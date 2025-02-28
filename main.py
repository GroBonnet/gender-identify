from utils import show_pictures
from preprocessing import select_data, load_data_RFC, split_data_DNN
from methods import GenderRFClassifier, GenderDNNClassifier

def main():
    # clf = GenderRFClassifier(image_size=(128, 128))

    # X_train, X_test, y_train, y_test = clf.process_data(method='test')

    # clf.train(X_train, y_train)

    # clf.evaluate(X_test, y_test)
    
    DNN = GenderDNNClassifier(image_size=(128, 128), lr=0.001, num_epochs=10, batch_size=32)
    DNN.process_data(method='run')
    DNN.train()
    DNN.evaluate()


if __name__ == '__main__':
    main()