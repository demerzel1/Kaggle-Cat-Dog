import shutil
import random
import os

def main():
    data_dir = '/Users/demerzel/PycharmProjects/cat-dog/data/train'
    img_list = os.listdir(data_dir)
    train_cat_list = []
    train_dog_list = []
    for img in img_list:
        if img.split('.')[0] == 'cat':
            train_cat_list.append(os.path.join(data_dir, img))
        else:
            train_dog_list.append(os.path.join(data_dir, img))
    num_valid = 32
    valid_cat_list = train_cat_list[:num_valid]
    train_cat_list = train_cat_list[num_valid:]
    valid_dog_list = train_dog_list[:num_valid]
    train_dog_list = train_dog_list[num_valid:]
    
    train_list = train_cat_list + train_dog_list
    valid_list = valid_cat_list + valid_dog_list
    
    random.shuffle(train_list)
    random.shuffle(valid_list)

    with open('/Users/demerzel/PycharmProjects/cat-dog/data/train.txt', 'w') as f:
        content = ''
        for img in train_list:
            content =  content + img + '\n'
        f.writelines(content)
    with open('/Users/demerzel/PycharmProjects/cat-dog/data/valid.txt', 'w') as f:
        content = ''
        for img in valid_list:
            content =  content + img + '\n'
        f.writelines(content)
    
    test_dir = '/Users/demerzel/PycharmProjects/cat-dog/data/test'
    test_list = []
    for num in sorted([int(x.split('.')[0]) for x in os.listdir(test_dir)]):
        test_list.append(os.path.join(test_dir, str(num) + '.jpg'))
    with open('/Users/demerzel/PycharmProjects/cat-dog/data/test.txt', 'w') as f:
        content = ''
        for img in test_list:
            content =  content + img + '\n'
        f.writelines(content)


if __name__ == '__main__':
    main()