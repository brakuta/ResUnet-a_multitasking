import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image

from ResUnet_a.model import Resunet_a
#from ResUnet_a.model2 import Resunet_a2
from multitasking_utils import Tanimoto_dual_loss
import argparse
import os

from skimage.util.shape import view_as_windows
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import gc
import psutil
import ast
from prettytable import PrettyTable
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy


def extract_patches_test(binary_img_test_ref, patch_size):
    # Extract training patches
    stride = patch_size

    height, width = binary_img_test_ref.shape
    #print(height, width)

    num_patches_h = int(height / stride)
    num_patches_w = int(width / stride)
    #print(num_patches_h, num_patches_w)

    new_shape = (num_patches_h*num_patches_w, patch_size, patch_size)
    new_img_ref = np.zeros(new_shape)
    print(new_img_ref.shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        #columns
        for w in range(num_patches_w):
            new_img_ref[cont] = binary_img_test_ref[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)

    return new_img_ref


def extract_patches_train(img_test_normalized, patch_size):
    # Extract training patches manual
    stride = patch_size

    height, width, channel = img_test_normalized.shape
    #print(height, width)

    num_patches_h = height // stride
    num_patches_w = width // stride
    #print(num_patches_h, num_patches_w)

    new_shape = (num_patches_h*num_patches_w, patch_size, patch_size, channel)
    new_img = np.zeros(new_shape)
    print(new_img.shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        # columns
        for w in range(num_patches_w):
            new_img[cont] = img_test_normalized[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)


    return new_img


def Test(model, patch_test, args):
    result = model.predict(patch_test)
    if args.multitasking:
        predicted_class = np.argmax(result[0], axis=-1)
    else:
        predicted_class = np.argmax(result, axis=-1)
    return predicted_class


def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    #avg_accuracy = 100*accuracy_score(true_labels, predicted_labels, average=None)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision


def compute_accuracy(labels, preds):
    true_labels = np.reshape(labels, (labels.shape[0] * labels.shape[1] * labels.shape[2]))
    predicted_labels = np.reshape(preds, (preds.shape[0] * preds.shape[1] * preds.shape[2]))
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


def add_tensorboard_scalars(train_writer, val_writer, epoch,
                            metric_name, train_loss, val_loss,
                            train_acc=None, val_acc=None):
    with train_writer.as_default():
        tf.summary.scalar(metric_name+'/Loss', train_loss,
                          step=epoch)
        if train_acc is not None:
            tf.summary.scalar(metric_name+'/Accuracy', train_acc,
                              step=epoch)
    with val_writer.as_default():
        tf.summary.scalar(metric_name+'/Loss', val_loss,
                          step=epoch)
        if val_acc is not None:
            tf.summary.scalar(metric_name+'/Accuracy', val_acc,
                              step=epoch)


def test_on_batch(net, optimizer, loss, x_val_b, y_val_h_b_seg):
    val_logits = net(x_val_b, training=False)
    with tf.device("CPU:0"):
        logits_npy = val_logits.numpy().copy()
        preds = np.argmax(logits_npy, axis=-1)
        label_preds = np.argmax(y_val_h_b_seg, axis=-1)
    acc_batch = compute_accuracy(label_preds, preds)
    # print(f'Val logits: {val_logits.shape}')
    # print(type(val_logits))
    loss_value = loss(y_val_h_b_seg, val_logits)
    return loss_value, acc_batch


def train_on_batch(net, optimizer, loss, x_train_b, y_train_h_b_seg):
    with tf.GradientTape() as tape:
        # Logits for this minibatch
        logits = net(x_train_b, training=True)
        # print('='*30 + ' [CHECKING LOSS] ' + '='*30)
        # print(f'Train logits: {logits.shape}')
        # print(type(logits))
        # print(logits.shape)
        # print(y_train_h_b_seg.shape)
        with tf.device("CPU:0"):
            logits_npy = logits.numpy().copy()
            preds = np.argmax(logits_npy, axis=-1)
            label_preds = np.argmax(y_train_h_b_seg, axis=-1)
        # print(preds.shape)
        # print(label_preds.shape)
        acc_batch = compute_accuracy(label_preds, preds)
        # print(acc_batch)

        # Compute the loss value for this minibatch.
        loss_value = loss(y_train_h_b_seg, logits)
        # print(type(loss_value))
        # print(loss_value.shape)
        # print(loss_value)
        # print(float(loss_value))

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable
    # variables with respect to the loss.
    grads = tape.gradient(loss_value, net.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, acc_batch


def train_model(args, net, x_train_paths, y_train_paths, x_val_paths,
                y_val_paths, batch_size, epochs, optimizer, loss,
                x_shape_batch, y_shape_batch,
                patience=10, delta=0.001):
    # patches_train = x_train_paths
    print('Start training...')
    print('='*60)
    print(f'Training on {len(x_train_paths)} images')
    print(f'Validating on {len(x_val_paths)} images')
    print('='*60)
    print(f'Total Epochs: {epochs}')
    # Initialize tensorboard metrics
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_path, 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_path, 'val'))
    # Initialize as maximum possible number
    min_loss = float('inf')
    cont = 0
    x_train_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_train_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    x_val_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_val_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)


    if args.multitasking:
        # Bounds
        if args.bound:
            y_train_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
            y_val_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
        # Dists
        if args.dist:
            y_train_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
            y_val_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
        # Colors
        if args.color:
            y_train_h_b_color = np.zeros((y_shape_batch[0],
                                          y_shape_batch[1],
                                          y_shape_batch[2], 3),
                                         dtype=np.float32)
            y_val_h_b_color = np.zeros((y_shape_batch[0],
                                        y_shape_batch[1],
                                        y_shape_batch[2], 3),
                                       dtype=np.float32)
    print(net.metrics_names)
    for epoch in range(epochs):
        if not args.multitasking:
            # DEBUG
            running_acc_tr = 0.0
            running_acc_val = 0.0
            running_loss_val = []
            running_loss_tr = []
            loss_tr = np.zeros((1, 2), dtype=np.float32)
            loss_val = np.zeros((1, 2), dtype=np.float32)
        else:
            metrics_len = len(net.metrics_names)
            loss_tr = np.zeros((1, metrics_len))
            loss_val = np.zeros((1, metrics_len))
        # Computing the number of batchs on training
        n_batchs_tr = len(x_train_paths)//batch_size
        # Random shuffle the data
        if not args.multitasking:
            (x_train_paths_rand,
             y_train_paths_rand_seg) = shuffle(x_train_paths, y_train_paths[0])
        else:
            (x_train_paths_rand, y_train_paths_rand_seg,
             y_train_paths_rand_bound, y_train_paths_rand_dist,
             y_train_paths_rand_color) \
             = shuffle(x_train_paths, y_train_paths[0], y_train_paths[1],
                       y_train_paths[2], y_train_paths[3])

        # Training the network per batch
        for batch in tqdm(range(n_batchs_tr), desc="Train"):
            x_train_paths_b = x_train_paths_rand[batch * batch_size:(batch + 1) * batch_size]
            y_train_paths_b_seg = y_train_paths_rand_seg[batch * batch_size:(batch + 1) * batch_size]
            if args.multitasking:
                y_train_paths_b_bound = y_train_paths_rand_bound[batch * batch_size:(batch + 1) * batch_size]

                y_train_paths_b_dist = y_train_paths_rand_dist[batch * batch_size:(batch + 1) * batch_size]

                y_train_paths_b_color = y_train_paths_rand_color[batch * batch_size:(batch + 1) * batch_size]
            # Load images in batch
            for b in range(batch_size):
                x_train_b[b] = np.load(x_train_paths_b[b])
                y_train_h_b_seg[b] = np.load(y_train_paths_b_seg[b])
                if args.multitasking:
                    if args.bound:
                        y_train_h_b_bound[b] = np.load(y_train_paths_b_bound[b])
                    if args.dist:
                        y_train_h_b_dist[b] = np.load(y_train_paths_b_dist[b])
                    if args.color:
                        y_train_h_b_color[b] = np.load(y_train_paths_b_color[b])

            if not args.multitasking:
                # loss_tr = loss_tr + net.train_on_batch(x_train_b, y_train_h_b_seg)
                loss_value, acc_tr = train_on_batch(net, optimizer, loss, x_train_b, y_train_h_b_seg)
                running_acc_tr += acc_tr * batch_size
                # Because loss is calculated as mean of batches
                # running_loss_tr += float(loss_value) * batch_size
                running_loss_tr.append(loss_value.numpy())
                #print(running_loss_tr)

            else:
                y_train_b = {"seg": y_train_h_b_seg}
                if args.bound:
                    y_train_b['bound'] = y_train_h_b_bound
                if args.dist:
                    y_train_b['dist'] = y_train_h_b_dist
                if args.color:
                    y_train_b['color'] = y_train_h_b_color

                loss_tr = loss_tr + net.train_on_batch(x=x_train_b, y=y_train_b)

            # print('='*30 + ' [CHECKING LOSS] ' + '='*30)
            # print(net.metrics_names)
            # print(type(loss_tr))
            # print(len(loss_tr))
            # print(loss_tr)
            # print(loss_tr.shape)

        # Training loss; Divide by the number of batches
        # print(loss_tr_debg)
        # loss_tr_float = running_loss_tr/len(x_train_paths)
        loss_tr_float = np.sum(running_loss_tr)/n_batchs_tr
        loss_tr = loss_tr/n_batchs_tr
        acc_tr = running_acc_tr/len(x_train_paths)

        # Computing the number of batchs on validation
        n_batchs_val = len(x_val_paths)//batch_size

        # Evaluating the model in the validation set
        for batch in tqdm(range(n_batchs_val), desc="Validation"):
            x_val_paths_b = x_val_paths[batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_seg = y_val_paths[0][batch * batch_size:(batch + 1) * batch_size]
            if args.multitasking:
                y_val_paths_b_bound = y_val_paths[1][batch * batch_size:(batch + 1) * batch_size]

                y_val_paths_b_dist = y_val_paths[2][batch * batch_size:(batch + 1) * batch_size]

                y_val_paths_b_color = y_val_paths[3][batch * batch_size:(batch + 1) * batch_size]
            # Load images in batch
            for b in range(batch_size):
                x_val_b[b] = np.load(x_val_paths_b[b])
                y_val_h_b_seg[b] = np.load(y_val_paths_b_seg[b])
                if args.multitasking:
                    if args.bound:
                        y_val_h_b_bound[b] = np.load(y_val_paths_b_bound[b])
                    if args.dist:
                        y_val_h_b_dist[b] = np.load(y_val_paths_b_dist[b])
                    if args.color:
                        y_val_h_b_color[b] = np.load(y_val_paths_b_color[b])

            if not args.multitasking:
                # loss_val = loss_val + net.test_on_batch(x_val_b, y_val_h_b_seg)
                loss_value, acc_val = test_on_batch(net, optimizer, loss, x_val_b, y_val_h_b_seg)
                running_acc_val += acc_val * batch_size
                # running_loss_val += float(loss_value) * batch_size
                running_loss_val.append(loss_value.numpy())
            else:
                # Dict template: y_val_b = {"segmentation": y_val_h_b_seg,
                # "boundary": y_val_h_b_bound, "distance":  y_val_h_b_dist,
                # "color": y_val_h_b_color}
                y_val_b = {"seg": y_val_h_b_seg}
                if args.bound:
                    y_val_b['bound'] = y_val_h_b_bound
                if args.dist:
                    y_val_b['dist'] = y_val_h_b_dist
                if args.color:
                    y_val_b['color'] = y_val_h_b_color

                loss_val = loss_val + net.test_on_batch(x=x_val_b, y=y_val_b)

        # loss_val_float = running_loss_val/len(x_val_paths)
        loss_val_float = np.sum(running_loss_val)/n_batchs_val
        loss_val = loss_val/n_batchs_val
        acc_val = running_acc_val/len(x_val_paths)

        if not args.multitasking:
            # DEBUG
            # train_loss = loss_tr[0, 0]
            train_acc = acc_tr
            # val_loss = loss_val[0, 0]
            val_acc = acc_val
            train_loss = loss_tr_float
            val_loss = loss_val_float
            print(f"Epoch: {epoch}" +
                  f" Training loss: {train_loss :.5f}" +
                  f" Train acc.: {100*train_acc:.5f}%" +
                  f" Validation loss: {val_loss :.5f}" +
                  f" Validation acc.: {100*val_acc:.5f}%")

            add_tensorboard_scalars(train_summary_writer, val_summary_writer,
                                    epoch, 'Total', train_loss, val_loss,
                                    train_acc, val_acc)
        else:
            train_metrics = dict(zip(net.metrics_names, loss_tr.tolist()[0]))
            # print(loss_tr.tolist()[0])
            val_metrics = dict(zip(net.metrics_names, loss_val.tolist()[0]))

            metrics_table = PrettyTable()
            metrics_table.title = f'Epoch: {epoch}'
            metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                         'Acc %', 'Val Acc %']
            metrics_table.add_row(['Seg', round(train_metrics['seg_loss'], 5),
                                  round(val_metrics['seg_loss'], 5),
                                  round(100*train_metrics['seg_accuracy'], 5),
                                  round(100*val_metrics['seg_accuracy'], 5)])

            add_tensorboard_scalars(train_summary_writer, val_summary_writer,
                                    epoch, 'Segmentation',
                                    train_metrics['seg_loss'],
                                    val_metrics['seg_loss'],
                                    train_metrics['seg_accuracy'],
                                    val_metrics['seg_accuracy'])

            if args.bound:
                metrics_table.add_row(['Bound',
                                       round(train_metrics['bound_loss'], 5),
                                      round(val_metrics['bound_loss'], 5),
                                      round(100*train_metrics['bound_accuracy'], 5),
                                      round(100*val_metrics['bound_accuracy'], 5)])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Boundary',
                                        train_metrics['bound_loss'],
                                        val_metrics['bound_loss'],
                                        train_metrics['bound_accuracy'],
                                        val_metrics['bound_accuracy'])
            if args.dist:
                metrics_table.add_row(['Dist',
                                       round(train_metrics['dist_loss'], 5),
                                       round(val_metrics['dist_loss'], 5),
                                       0, 0])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Distance',
                                        train_metrics['dist_loss'],
                                        val_metrics['dist_loss'])
            if args.color:
                metrics_table.add_row(['Color',
                                       round(train_metrics['color_loss'], 5),
                                       round(val_metrics['color_loss'], 5),
                                       0, 0])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Color',
                                        train_metrics['color_loss'],
                                        val_metrics['color_loss'])

            metrics_table.add_row(['Total', round(train_metrics['loss'], 5),
                                  round(val_metrics['loss'], 5),
                                  0, 0])

            add_tensorboard_scalars(train_summary_writer,
                                    val_summary_writer,
                                    epoch, 'Total',
                                    train_metrics['loss'],
                                    val_metrics['loss'])
            val_loss = val_metrics['loss']
            print(metrics_table)
        # Early stop
        # Save the model when loss is minimum
        # Stop the training if the loss don't decreases after patience epochs
        if val_loss >= min_loss + delta:
            cont += 1
            print(f'EarlyStopping counter: {cont} out of {patience}')
            if cont >= patience:
                print("Early Stopping! \t Training Stopped")
                print("Saving model...")
                net.save('weights/model_early_stopping.h5')
                return 0
        else:
            cont = 0
            # best_score = score
            min_loss = val_loss
            print("Saving best model...")
            net.save('weights/best_model.h5')

# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resunet_a", help="choose resunet-a model or not",
                        type=int, default=0)
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", type=int, default=0)
    parser.add_argument("--bound", help="choose resunet-a boundary task or not",
                        type=int, default=1)
    parser.add_argument("--dist", help="choose resunet-a distance task or not",
                        type=int, default=1)
    parser.add_argument("--color", help="choose resunet-a color task or not",
                        type=int, default=1)
    parser.add_argument("--gpu_parallel",
                        help="choose 1 to train one multiple gpu",
                        type=int, default=0)
    parser.add_argument("--log_path", help="path where to save logs",
                        type=str, default='./results/log_run1')
    parser.add_argument("--dataset_path", help="path where to load dataset",
                        type=str, default='./DATASETS/patches_ps=256_stride=32')
    parser.add_argument("--batch_size", help="batch size on training",
                        type=int, default=4)
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate on training",
                        type=float, default=1e-3)
    parser.add_argument("--loss", help="choose which loss you want to use",
                        type=str, default='weighted_cross_entropy',
                        choices=['weighted_cross_entropy', 'cross_entropy',
                                 'tanimoto'])
    args = parser.parse_args()

    if args.gpu_parallel:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = None

    # Load images

    root_path = args.dataset_path
    train_path = os.path.join(root_path, 'train')
    patches_tr = [os.path.join(train_path, name)
                  for name in os.listdir(train_path)]

    ref_path = os.path.join(root_path, 'labels/seg')
    patches_tr_lb_h = [os.path.join(ref_path, name) for name
                       in os.listdir(ref_path)]

    if args.multitasking:
        ref_bound_path = os.path.join(root_path, 'labels/bound')
        print(ref_bound_path)
        patches_bound_labels = [os.path.join(ref_bound_path, name) for name
                                in os.listdir(ref_bound_path)]

        ref_dist_path = os.path.join(root_path, 'labels/dist')
        patches_dist_labels = [os.path.join(ref_dist_path, name) for name
                               in os.listdir(ref_dist_path)]

        ref_color_path = os.path.join(root_path, 'labels/color')
        patches_color_labels = [os.path.join(ref_color_path, name) for name
                                in os.listdir(ref_color_path)]

    if args.multitasking:
        patches_tr, patches_val, patches_tr_lb_h, patches_val_lb_h, patches_bound_labels_tr, patches_bound_labels_val, patches_dist_labels_tr, patches_dist_labels_val, patches_color_labels_tr, patches_color_labels_val   = train_test_split(patches_tr, patches_tr_lb_h, patches_bound_labels, patches_dist_labels, patches_color_labels,  test_size=0.2, random_state=42)
    else:
        patches_tr, patches_val, patches_tr_lb_h, patches_val_lb_h = train_test_split(patches_tr, patches_tr_lb_h, test_size=0.2, random_state=42)

    number_class = 5
    patch_size = 256
    # stride = patch_size // 8
    epochs = 500

    if args.multitasking:
        '''
            index maps:
                0 --> segmentation
                1 --> boundary
                2 --> distance
                3 --> color
        '''
        y_paths = [patches_tr_lb_h, patches_bound_labels_tr,
                   patches_dist_labels_tr, patches_color_labels_tr]

        val_paths = [patches_val_lb_h, patches_bound_labels_val,
                     patches_dist_labels_val, patches_color_labels_val]
    else:
        y_paths = [patches_tr_lb_h]

        val_paths = [patches_val_lb_h]

    rows = patch_size
    cols = patch_size
    channels = 3
    adam = Adam(lr=args.learning_rate, beta_1=0.9)
    sgd = SGD(lr=args.learning_rate, momentum=0.8)

    print(f'Using {args.loss} loss!')
    if args.loss == 'cross_entropy':
        loss = CategoricalCrossentropy(from_logits=True)
        # loss = "categorical_crossentropy"
    elif args.loss == 'weighted_cross_entropy':
        weights = [4.34558461, 2.97682037, 3.92124661, 5.67350328, 374.0300152]
        print('='*60)
        print(weights)
        loss = weighted_categorical_crossentropy(weights)
    elif args.loss == 'tanimoto':
        loss = Tanimoto_dual_loss()

    if args.resunet_a:
        if args.multitasking:
            print('Multitasking enabled!')
            tanimoto = Tanimoto_dual_loss()
            resuneta = Resunet_a((rows, cols, channels), number_class, args)
            model = resuneta.model
            model.summary()

            losses = {'seg': tanimoto}
            lossWeights = {'seg': 1.0}
            if args.bound:
                losses['bound'] = tanimoto
                lossWeights["bound"] = 1.0
            if args.dist:
                losses['dist'] = tanimoto
                lossWeights["dist"] = 1.0
            if args.color:
                losses['color'] = tanimoto
                lossWeights["color"] = 1.0

            print(losses)
            print(lossWeights)
            if args.gpu_parallel:
                with strategy.scope():
                    model.compile(optimizer=adam, loss=losses,
                                  loss_weights=lossWeights,
                                  metrics=['accuracy'])
            else:
                model.compile(optimizer=adam, loss=losses,
                              loss_weights=lossWeights, metrics=['accuracy'])
        else:
            resuneta = Resunet_a((rows, cols, channels), number_class, args)
            model = resuneta.model
            model.summary()
            # model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

        print('ResUnet-a compiled!')
    else:
        model = unet((rows, cols, channels), number_class)
        model.summary()

        # model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    filepath = './models/'

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # train the model
    if args.multitasking:
        x_shape_batch = (args.batch_size, patch_size, patch_size, 3)
        y_shape_batch = (args.batch_size, patch_size, patch_size, 5)
        start_time = time.time()
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)
        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')
    else:
        x_shape_batch = (args.batch_size, patch_size, patch_size, 3)
        y_shape_batch = (args.batch_size, patch_size, patch_size, 5)

        start_time = time.time()

        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, epochs, adam, loss,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)

        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')

    # #%% Test model
    #
    # # Load images
    # img_test_path = 'Image_Test.npy'
    # img_test = load_npy_image(os.path.join(root_path, img_test_path)).astype(np.float32)
    # # Normalizes the image
    # img_test_normalized = normalization(img_test)
    # # Transform the image into W x H x C shape
    # img_test_normalized = img_test_normalized.transpose((1,2,0))
    # print(img_test_normalized.shape)
    #
    # # Load reference
    # img_test_ref_path = 'Reference_Test.npy'
    # img_test_ref = load_npy_image(os.path.join(root_path, img_test_ref_path))
    # img_test_ref = img_test_ref.transpose((1,2,0))
    # print(img_test_ref.shape)
    #
    # # Create binarized matrix
    # w = img_test_ref.shape[0]
    # h = img_test_ref.shape[1]
    # c = img_test_ref.shape[2]
    # # binary_img_train_ref = np.zeros((1,w,h))
    # binary_img_test_ref = np.full((w, h), -1)
    # # Dictionary used in training
    # label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}
    # # label = 0
    # for i in range(w):
    #     for j in range(h):
    #         r = img_test_ref[i][j][0]
    #         g = img_test_ref[i][j][1]
    #         b = img_test_ref[i][j][2]
    #         rgb = (r,g,b)
    #         rgb_key = str(rgb)
    #         binary_img_test_ref[i][j] = label_dict[rgb_key]
    # print(label_dict)
    #
    # # Put the patch size according to you training here
    # patches_test = extract_patches_train(img_test_normalized, patch_size)
    # patches_test_ref = extract_patches_test(binary_img_test_ref, patch_size)
    #
    # #% Load model
    # model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
    # # Prediction
    # # Test the model
    # patches_pred = Test(model, patches_test, args)
    # print(patches_pred.shape)
    #
    # # Metrics
    # true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0]* patches_test_ref.shape[1]*patches_test_ref.shape[2]))
    # predicted_labels = np.reshape(patches_pred, (patches_pred.shape[0]* patches_pred.shape[1]*patches_pred.shape[2]))
    #
    # # Metrics
    # metrics = compute_metrics(true_labels,predicted_labels)
    # cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4])
    #
    # print('Confusion  matrix \n', cm)
    # print()
    # print('Accuracy: ', metrics[0])
    # print('F1score: ', metrics[1])
    # print('Recall: ', metrics[2])
    # print('Precision: ', metrics[3])
    #
    # def pred_recostruction(patch_size, pred_labels, binary_img_test_ref):
    #     # Patches Reconstruction
    #     stride = patch_size
    #
    #     height, width = binary_img_test_ref.shape
    #
    #     num_patches_h = height // stride
    #     num_patches_w = width // stride
    #
    #     new_shape = (height, width)
    #     img_reconstructed = np.zeros(new_shape)
    #     cont = 0
    #     # rows
    #     for h in range(num_patches_h):
    #         # columns
    #         for w in range(num_patches_w):
    #             img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = patches_pred[cont]
    #             cont += 1
    #     print('Reconstruction Done!')
    #     return img_reconstructed
    #
    #
    # def reconstruction_rgb_prdiction_patches(img_reconstructed, label_dict):
    #     reversed_label_dict = {value : key for (key, value) in label_dict.items()}
    #     print(reversed_label_dict)
    #     height, width = img_reconstructed.shape
    #     img_reconstructed_rgb = np.zeros((height,width,3))
    #     for h in range(height):
    #         for w in range(width):
    #             pixel_class = img_reconstructed[h, w]
    #             img_reconstructed_rgb[h, w, :] = ast.literal_eval(reversed_label_dict[pixel_class])
    #     print('Conversion to RGB Done!')
    #     return img_reconstructed_rgb.astype(np.uint8)
    #
    # img_reconstructed = pred_recostruction(patch_size, patches_pred, binary_img_test_ref)
    # img_reconstructed_rgb = reconstruction_rgb_prdiction_patches(img_reconstructed, label_dict)
    #
    # plt.imsave(f'img_reconstructed_rgb_exp{exp}.jpeg', img_reconstructed_rgb)
