# 最初针对手写数字识别，使用的是输入32*32的图片，

def LeNet_5():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), strides=(1, 1), input_shape=(32, 32, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5),strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model