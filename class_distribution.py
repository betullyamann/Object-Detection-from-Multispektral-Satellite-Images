import numpy as np

def count_elements(labels, pixels, idx):
    bins = [[], [], [], [], [], [], [], [], [], [], []]
    bins_pixels = [[], [], [], [], [], [], [], [], [], [], []]
    bins_0 =[]
    bins_p0 = []
    bins_1 = []
    bins_p1 = []
    bins_2 = []
    bins_p2 = []
    bins_3 = []
    bins_p3 = []
    bins_4 = []
    bins_p4 = []
    bins_5 = []
    bins_p5 = []
    bins_6 = []
    bins_p6 = []
    bins_7 = []
    bins_p7 = []
    bins_8 = []
    bins_p8 = []
    bins_9 = []
    bins_p9 = []
    bins_10 =[]
    bins_p10 = []


    for label, pixel, i in zip(labels, pixels, range(len(labels))):
        sum = np.sum(label)
        if sum == 0:
            bins[10].append(np.array([0,0,0,0,0,0,0,0,0,0,0]))
            bins_pixels[10].append(pixel)
        elif sum == 1:
            tmp = np.array([0,0,0,0,0,0,0,0,0,0,0])
            tmp[np.argmax(label)] = 1
            bins[np.argmax(label)].append(tmp)
            bins_pixels[np.argmax(label)].append(pixel)

    bins[0] = np.array(bins[0])
    print(bins[0].shape)

    arr2 = [1, 2, 6]

    tmp = []
    tmp_b =[]


    for k, l in zip(bins[1], bins_pixels[1]):
        for m in range(5):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_1.append(tmp[k])
        bins_p1.append(tmp_b[k])

    tmp = []
    tmp_b = []
    for k, l in zip(bins[2], bins_pixels[2]):
        for m in range(5):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_2.append(tmp[k])
        bins_p2.append(tmp_b[k])

    tmp = []
    tmp_b = []
    for k, l in zip(bins[6], bins_pixels[6]):
        for m in range(5):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_6.append(tmp[k])
        bins_p6.append(tmp_b[k])

    arr3 = [7, 8, 9]

    tmp = []
    tmp_b = []
    for k, l in zip(bins[7], bins_pixels[7]):
        for m in range(10):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_7.append(tmp[k])
        bins_p7.append(tmp_b[k])

    tmp = []
    tmp_b = []
    for k, l in zip(bins[8], bins_pixels[8]):
        for m in range(1000):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_8.append(tmp[k])
        bins_p8.append(tmp_b[k])

    tmp = []
    tmp_b = []
    for k, l in zip(bins[9], bins_pixels[9]):
        for m in range(200):
            tmp.append(k)
            tmp_b.append(l)

    for k in range(100000):
        bins_9.append(tmp[k])
        bins_p9.append(tmp_b[k])

    for k in range(100000):
        bins_0.append(bins[0][k])
        bins_p0.append(bins_pixels[0][k])

    for k in range(100000):
        bins_3.append(bins[3][k])
        bins_p3.append(bins_pixels[3][k])

    for k in range(100000):
        bins_4.append(bins[4][k])
        bins_p4.append(bins_pixels[4][k])

    for k in range(100000):
        bins_5.append(bins[5][k])
        bins_p5.append(bins_pixels[5][k])

    for k in range(100000):
        bins_10.append(bins[10][k])
        bins_p10.append(bins_pixels[10][k])

    final_labels = []
    for i in bins_0:
        final_labels.append(i)
    for i in bins_1:
        final_labels.append(i)
    for i in bins_2:
        final_labels.append(i)
    for i in bins_3:
        final_labels.append(i)
    for i in bins_4:
        final_labels.append(i)
    for i in bins_5:
        final_labels.append(i)
    for i in bins_6:
        final_labels.append(i)
    for i in bins_7:
        final_labels.append(i)
    for i in bins_8:
        final_labels.append(i)
    for i in bins_9:
        final_labels.append(i)
    for i in bins_10:
        final_labels.append(i)


    final_pixels = []
    for i in bins_p0:
        final_pixels.append(i)
    for i in bins_p1:
        final_pixels.append(i)
    for i in bins_p2:
        final_pixels.append(i)
    for i in bins_p3:
        final_pixels.append(i)
    for i in bins_p4:
        final_pixels.append(i)
    for i in bins_p5:
        final_pixels.append(i)
    for i in bins_p6:
        final_pixels.append(i)
    for i in bins_p7:
        final_pixels.append(i)
    for i in bins_p8:
        final_pixels.append(i)
    for i in bins_p9:
        final_pixels.append(i)
    for i in bins_p10:
        final_pixels.append(i)

    print(np.mean(np.array(final_labels)))
    print(np.mean(np.array(final_labels[0])))
    print(np.array(final_labels).shape)
    print(np.array(final_pixels).shape)

    np.save("final_labels.npy", final_labels)
    np.save("final_pixels" + str(idx) + ".npy", final_pixels)


if __name__ == '__main__':
    labels = np.load("labels.npy")
    conv = [1, 3, 5, 7, 9, 11, 13]
    for i in conv:
        pixels = np.load("pixels_" + str(i) + ".npy")
        count_elements(labels, pixels, i)


