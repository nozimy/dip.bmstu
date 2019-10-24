from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Каталог с данными для обучения, проверки, тестирования
train_dir = 'Training'
val_dir = 'Training'
test_dir = 'Testing'
# Размеры изображения
img_width, img_height = 200, 200
# Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

epochs = 20  # Количество эпох
batch_size = 2  # Размер мини-выборки
nb_train_samples = 40  # Количество изображений для обучения
nb_validation_samples = 40  # Количество изображений для проверки
nb_test_samples = 10  # Количество изображений для тестирования


#Наша сеть будет состоять из трех слоев Convolution_2D и слоев MaxPooling2D 
#после каждой свертки. После этого выходное изображение слоя подвыборки 
#трансформируется в одномерный вектор (слоем Flatten) и проходит два полносвязных 
#слоя (Dense). На всех слоях, кроме выходного полносвязного слоя, используется
#функция активации ReLU, последний же слой использует sigmoid.
#Для регуляризации нашей модели после первого 
#полносвязного слоя применяется слой Dropout. Здесь Keras также выделяется 
#на фоне остальных фреймворков: в нем есть внутренний флаг, который автоматически 
#включает и выключает dropout, в зависимости от того, находится модель в фазе 
#обучения или тестирования.
def create_nn():
#    В Keras мы используем слои для построения моделей. Обычно модель - это граф,
#    состоящий из нескольких слоев. Самый распространенный тип модели это стэк 
#    идущих друг за другом слоев - последовательная модель tf.keras.Sequential.
    model = Sequential()

#    (3,3) - Шаг (stride) — на сколько смещается ядро на каждом шаге при 
#    вычислении следующего пикселя результирующего изображения. Обычно его
#    принимают равным 1, и чем больше его значение, тем меньше размер 
#    выходного изображения;
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#    activation: Устанавливает функцию активации для данного слоя. Этот параметр должен 
#    указываться в имени встроенной функции или использоваться как вызываемый 
#    объект. По умолчанию активация не используется
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

#    После этого выходное изображение слоя подвыборки трансформируется в 
#    одномерный вектор (слоем Flatten) и проходит два полносвязных слоя (Dense).
    model.add(Flatten())
    # Добавим в нашу модель слой `Dense` из 64 блоков:
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Создаем сигмоидный слой:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #Мы используем перекрестную энтропию в качестве функции потерь;
    #Мы используем оптимизатор Адама для градиентного спуска;
    #Мы измеряем точность модели (так как исходные данные распределены по классам равномерно);
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def create_generator(gen, dir_name):
#   flow_from_directory: Takes the path to a directory & generates batches of augmented data.
    generator = gen.flow_from_directory(
        dir_name,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    return generator


def fit_nn(loc_model, train_generator, val_generator):
#    Trains the model on data generated batch-by-batch by a Python generator 
    loc_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)
    return loc_model


def get_data_and_label_from_gen(gen):
    x, y = zip(*(gen[i] for i in range(len(gen))))
    x_value, y_value = np.vstack(x), np.vstack(y)
    return x_value, y_value.reshape(-1)


def get_real_label(nlabel):
    return "Stanly Weber" if nlabel == 1 else "Sehun"

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = int(predictions_array[i]), int(true_label[i])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def show_graph(image, orig_label, predict_label):
    plt.figure(figsize=(10,10))
    for i, img in enumerate(image):
        if i < 2:
            plt.subplot(1, 2, i + 1)
            plt.imshow(img.reshape(img_width, img_height, 3), cmap='gray', interpolation='none')
            plt.title("Class: {} Predict: {}".format(orig_label[i], predict_label[i]))
            plt.xlabel(get_real_label(predict_label[i]))
    plt.tight_layout()
    plt.show()   

if __name__ == '__main__':

    # Create data generator
    data_gen = ImageDataGenerator(rescale=1. / 255)
    train_gen = create_generator(data_gen, train_dir)
    val_gen = create_generator(data_gen, val_dir)
    test_gen = create_generator(data_gen, test_dir)

    # Load or create nn
    try:
        model = load_model("model.h5py")
    except (OSError, ImportError, ValueError):
        model = create_nn()
        model = fit_nn(model, train_gen, val_gen)
        model.save("model.h5py")

    model.summary()

    # Predict
    test_x, test_y = get_data_and_label_from_gen(test_gen)
    predict = np.round(model.predict(test_x, batch_size=batch_size)).reshape(-1)
    print("Исходная разметка: {} \nПредсказананная: {}".format(test_y, predict))
        
        
    # Show results
    scores = model.evaluate_generator(test_gen)
    print("Точность: %.2f%%" % (scores[1] * 100))
    show_graph(test_x, test_y, predict)
    

