import numpy as np

def count_elements(labels, pixels):
    bins = [[], [], [], [], [], [], [], [], [], [], []]
    bins_pixels = [[], [], [], [], [], [], [], [], [], [], []]

    for label, pixel, i in zip(labels, pixels, range(len(labels))):
        sum = np.sum(label)
        if sum == 0:
            bins[10].append(label)
            bins_pixels[10].append(pixel)
        elif sum == 1:
            bins[np.argmax(label)].append(label)
            bins_pixels[np.argmax(label)].append(pixel)

    bins[0] = np.array(bins[0])
    print(bins[0].shape)

    arr = [0, 3, 4, 5, 10]
    for j in arr:
        bins[j] = bins[j, :100000]
        bins_pixels[j] = bins_pixels[j, :100000]

    arr2 = [1, 2, 6]
    for i in arr2:
        for j in range(3):
            bins[i].extend(bins[i])
            bins_pixels[i].extend(bins_pixels[i])

    arr3 = [7, 8, 9]
    for i in arr3:
        for j in range(3):
            bins[i].extend(bins[i])
            bins_pixels[i].extend(bins_pixels[i])


def cdist_main():
    conv = [1, 3, 5, 7, 9, 11, 13]
    labels = np.load("labels.npy")
    for i in conv:
        pixels = np.load("pixels_" + str(i) + ".npy")

        count_elements(labels)
        new_pixels, new_labels = balance_class_distribution(labels, pixels)

        new_pixels = np.array(new_pixels)
        new_labels = np.array(new_labels)

        n_pxls = []
        n_labels = []
        idxs = np.random.choice(len(new_pixels), 200000)

        for j in idxs:
            n_pxls.append(new_pixels[j])
            n_labels.append(new_labels[j])

        n_pxls = np.array(n_pxls)
        n_labels = np.array(n_labels)

        np.save("selected_pixels_" + str(i), np.array(n_pxls))
        np.save("selected_labels", np.array(n_labels))


if __name__ == '__main__':
    labels = np.load("labels.npy")
    pixels = np.load("pixels_1.npy")
    count_elements(labels, pixels)
   # cdist_main()

