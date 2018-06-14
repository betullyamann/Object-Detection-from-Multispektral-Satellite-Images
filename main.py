


h = 100
w = int(h*1.014)

if __name__ == '__main__':
    svm_kernels = ["poly", "linear", "sigmoid"]
    k_neighbors = [3, 5, 7]
    hidden_layer = [(20, 20, 20, 20), (40, 40, 40, 40)]

    print("NB")
    predictions = classification_with_naive_bayes()
    calculate_metrics("NB", predictions)