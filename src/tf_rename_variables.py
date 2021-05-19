"""
Modifications copyright (C) 2020 Michael Strobl

Original script from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96

"""

import sys, getopt

import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ --model_checkpoint_path=path/to/dir/new_model_checkpoint ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, model_checkpoint_path, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
                new_name = new_name.lower()
                new_name = new_name.replace('matrix','kernel')
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)


        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, model_checkpoint_path)


def main(argv):
    checkpoint_dir = None
    model_checkpoint_path = None
    replace_from = 'figer_model/context_encoder/context_encoder/left_encoder/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'
    replace_to = 'figer_model/context_encoder/context_encoder/left_encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'model_checkpoint_path=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--model_checkpoint_path':
            model_checkpoint_path = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir or not model_checkpoint_path:
        print(model_checkpoint_path)
        print('Please specify a checkpoint_dir and new checkpoint path. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, model_checkpoint_path, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])
