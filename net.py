import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_dataset():
    # Путь к исходным изображениям
    data_path = 'd:\\NN\data'

    # Две последовательные трансформации- превращение изображение в тензор pytorch и нормализация
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Разделяем по заданному соотношению данные для обучения и проверки
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    train_len = int(len(train_dataset) * 0.95)
    train, eval = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    # Создаём загрузчики данных для обучения и проверки
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=128,
        num_workers=0,
        shuffle=True
    )

    eval_loader = torch.utils.data.DataLoader(
        eval,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    return train_loader, eval_loader

# Создаём класс сети, наследуя модуль из библиотеки
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        '''
        Задаём 6 свёрточных слоёв
        В первом ядро свёртки 5x5, в остальных 3x3
        Т.к. изображение цветное, то на вход первого слоя получаем 3 канала
        '''

        self.conv1 = nn.Conv2d(3, 36, 5)
        self.conv2 = nn.Conv2d(36, 36, 5)
        self.conv3 = nn.Conv2d(36, 36, 3)
        self.conv4 = nn.Conv2d(36, 12, 3)
        self.conv5 = nn.Conv2d(12, 6, 3)
        self.conv6 = nn.Conv2d(6, 6, 3)

        # Задаём полностью связанные слои
        self.fc1 = nn.Linear(96, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)

        # Задаём слои dropout 0.5 и 0.1 ,  max pooling 2x2
        self.drop05 = nn.Dropout(p=0.5)
        self.drop01 = nn.Dropout(p=0.1)
        self.pool = nn.MaxPool2d(2, 2)

        # 2d-нормализация для свёрточных слоёв
        self.norm1 = nn.BatchNorm2d(36)
        self.norm2 = nn.BatchNorm2d(36)
        self.norm3 = nn.BatchNorm2d(36)
        self.norm4 = nn.BatchNorm2d(12)
        self.norm5 = nn.BatchNorm2d(6)
        self.norm6 = nn.BatchNorm2d(6)

        # Одномерная нормализация для полностью связанных слоёв
        self.norm7 = nn.BatchNorm1d(32)
        self.norm8 = nn.BatchNorm1d(16)
        self.norm9 = nn.BatchNorm1d(8)

    def forward(self, x):

        # Последовательно пропускаем входные данные через все слои
        # Используется функция активации Elu

        x = self.norm1(F.elu(self.conv1(x)))
        x = self.norm2(F.elu(self.conv2(x)))
        x = self.pool(x)
        x = self.drop01(x)
        x = self.norm3(F.elu(self.conv3(x)))
        x = self.norm4(F.elu(self.conv4(x)))
        x = self.pool(x)
        x = self.drop01(x)
        x = self.norm5(F.elu(self.conv5(x)))
        x = self.norm6(F.elu(self.conv6(x)))

        # Преобразуем двумерный массив из свёрточных слоёв в одномерный
        x = x.view(-1, x.shape[1] * x.shape[3] * x.shape[2])

        x = self.norm7(F.elu(self.fc1(x)))
        x = self.drop05(x)
        x = self.norm8(F.elu(self.fc2(x)))
        x = self.drop05(x)
        x = self.norm9(F.elu(self.fc3(x)))
        x = self.drop05(x)
        x = self.fc4(x)
        return x

# Создаём экземпляр класса сети
net = Net()

# Задаём устройство для расчёта сети и переносим её на него.
device = torch.device("cuda:0")
net.to(device)

# Задаём функцию потерь и алгоритм оптимизации
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Загружаем данные с диска
dataset, evalset = load_dataset()

for epoch in range(1, 100):

    # Переводим сеть в режим обучения
    net.train()

    for data in dataset:
        # Разделяем входные данные сети и метки
        inputs, labels = data
        # И переносим их на устройство
        inputs, labels = inputs.to(device), labels.to(device)
        # Очищаем данные о предыдущих расчётах градиентов
        optimizer.zero_grad()
        # Отдаём сети входные данные, получаем результат
        outputs = net(inputs)
        # Оцениваем ошибку между полученным результатом и истинными значениями
        loss = criterion((outputs), labels)
        # Запускаем расчёт градиентов для метода обратного распространения ошибки
        loss.backward()
        # Корректируем веса
        optimizer.step()

    # Переводим сеть в режим оценки
    net.eval()

    # Создаём массив для хранения ошибок
    losses = np.array(())

    for data in evalset:
        # Разделяем входные данные сети и метки
        inputs, labels = data
        # И переносим их на устройство
        inputs, labels = inputs.to(device), labels.to(device)
        # Отдаём сети входные данные, получаем результат
        outputs = net(inputs)
        # Оцениваем ошибку
        loss = criterion(outputs, labels)
        # Добавляем величину ошибки в массив ошибок
        losses = np.append(losses, loss.item())

    # Выводим медианную ошибку за эпоху и номер эпохи
    print(f'Медианная ошибка: {losses.mean():.6f} Эпоха: {epoch}')

    # Сохраняем сеть в файл после каждой эпохи для безопасной остановки обучения в любой момент
    torch.save(net, 'd:\\NN\models\\' + '1.pt')

print('End')
