import numpy as np

labels = np.load("labels.npy")
pixels = np.load("pixels_1.npy")

def count_elements():
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


def balance_class_distribution():
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


if __name__ == '__main__':
    count_elements()
    new_pixels, new_labels = balance_class_distribution()

    np.save("balanced_pixels", np.array(new_pixels))
    np.save("balanced_labels", np.array(new_labels))
