import os, shutil

org_dataset_dir = '/home/han/code/data/dogsvscats'
org_dataset_train_dir = os.path.join(org_dataset_dir, 'train')

base_dir = os.path.join(org_dataset_dir, 'small')
# os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
val_dir =  os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# os.mkdir(train_dir)
# os.mkdir(val_dir)
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

val_cats_dir = os.path.join(val_dir, 'cats')
val_dogs_dir = os.path.join(val_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# os.mkdir(train_cats_dir)
# os.mkdir(val_cats_dir)
# os.mkdir(test_cats_dir)

# os.mkdir(train_dogs_dir)
# os.mkdir(val_dogs_dir)
# os.mkdir(test_dogs_dir)

def copy_data(dir, beg, end):
    sub_dir = os.path.join(dir, 'cats')
    fnames = ['cat.{}.jpg'.format(i) for i in range(beg, end)]
    for fname in fnames:
        s = os.path.join(org_dataset_train_dir, fname)
        d = os.path.join(sub_dir, fname)
        shutil.copyfile(s, d)
    
    sub_dir = os.path.join(dir, 'dogs')
    fnames = ['dog.{}.jpg'.format(i) for i in range(beg, end)]
    for fname in fnames:
        s = os.path.join(org_dataset_train_dir, fname)
        d = os.path.join(sub_dir, fname)
        shutil.copyfile(s, d)


copy_data(train_dir, 0, 1000)
copy_data(val_dir, 1000, 1500)
copy_data(test_dir, 1500, 2000)