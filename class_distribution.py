import numpy as np

def count_elements(labels):
    bins = [[], [], [], [], [], [], [], [], [], [], []]
    number_of_elements = []

    for label, i in zip(labels, range(len(labels))):
        sum = np.sum(label)
        if sum == 0:
            bins[10].append(i)
        elif sum == 1:
            bins[np.argmax(label)].append(i)
    '''
    for i in range(len(bins)):
        print(len(bins[i]))
    '''


def balance_class_distribution(labels, pixels):
    new_pixels = []
    new_labels = []

    for label, pixel in zip(labels, pixels):
        new_pixels.append(pixel)
        new_labels.append(label)

    for label, pixel in zip(labels, pixels):
        sum = np.sum(label)
        if sum != 0:
            if np.argmax(label) == 8 or np.argmax(label) == 9:
                for j in range(100):
                    new_pixels.append(pixel)
                    new_labels.append(label)
    '''
    bins = [[], [], [], [], [], [], [], [], [], [], []]
    
    for label2, i in zip(new_labels, range(len(new_labels))):
        sum = np.sum(label2)
        if sum == 0:
            bins[10].append(i)
        elif sum == 1:
            bins[np.argmax(label2)].append(i)

    for i in range(len(bins)):
        print(len(bins[i]))
    '''
    return new_pixels, new_labels

def random_sampling(pixels, labels, j):
    n_pxls = []
    n_labels = []
    idxs = np.random.choice(len(pixels), 200000)
    for i in idxs:
        n_pxls.append(pixels[i])
        n_labels.append(labels[i])

    n_pxls = np.array(n_pxls)
    n_labels = np.array(n_labels)

    np.save("selected_pixels_" + str(j), np.array(n_pxls))
    np.save("selected_labels", np.array(n_labels))


def cdist_main():
    conv = [1, 3, 5, 7, 9, 11, 13]
    labels = np.load("labels.npy")
    for i in conv:
        pixels = np.load("pixels_" + str(i) + ".npy")

        count_elements(labels)
        new_pixels, new_labels = balance_class_distribution(labels, pixels)
        np.save("balanced_pixels_" + str(i), np.array(new_pixels))
        np.save("balanced_labels", np.array(new_labels))

        new_pixels = np.load("balanced_pixels_" + str(i) + ".npy")
        new_labels = np.load("balanced_labels.npy")

        random_sampling(new_pixels, new_labels, i)


'''
if __name__ == '__main__':
    cdist_main()  
'''
