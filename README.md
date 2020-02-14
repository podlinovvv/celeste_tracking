# Сeleste tracking

## О проекте:

Это учебный проект, его основной целью являлось изучение свёрточных нейронных сетей и фреймворка Pytorch. 
   
В рамках проекта с помощью нейросети реализовано отслеживание игрового персонажа в игре Celeste.  
Данная игра выбрана из-за примитивной графики и достаточно большого количества различных анимаций персонажа.

Использовался Python 3.7, фреймворк Pytorch, библиотеки NumPy, OpenCV.  

### Исходные данные: 
* Сделаны скриншоты персонажа и фона из игры. ~60 для фона и ~100 для персонажа. 

* Скриншоты разрезаны на большое количество фрагментов размером 50x50 пикселей 
	и разделены на две папки - с персонажем и без него.  

#### Примеры исходных скриншотов:
#### Фон:
![фон](https://github.com/podlinovvv/celeste_tracking/blob/master/img/2/111.png 'фон') 
![фон](https://github.com/podlinovvv/celeste_tracking/blob/master/img/2/222.png 'фон')


#### Персонаж:
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/1.png 'персонаж 1') 
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/2.png 'персонаж 2') 
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/3.png 'персонаж 3') 
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/4.png 'персонаж 4')
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/6.png 'персонаж 6') 
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/7.png 'персонаж 7')
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/9.png 'персонаж 9') 
![персонаж](https://github.com/podlinovvv/celeste_tracking/blob/master/img/1/10.png 'персонаж 10') 


### Нейросеть:

Необходимо определить, на каком изображении присутствует персонаж, а на каком нет.  
С использованием фреймворка Pytorch была создана и обучена нейросеть для классификации изображений.  
Так как это учебный проект, то архитектура подбиралась самостоятельно.

#### Архитектура:
	2 свёрточных слоя с ядрами свёртки 5x5 и 3x3
	max pooling 2x2
	dropout 0.1
	2 свёрточных слоя с ядрами свёртки 3x3
	max pooling 2x2
	dropout 0.1
	2 свёрточных слоя с ядрами свёртки 3x3
	4 полностью связанных слоя с dropout 0.5 между ними

На выходе каждого слоя, кроме последнего, значения нормализованы batch normalization.  
В последнем слое использована функция softmax.  
В качестве функции активации использовалась Elu.  
Функция потерь - cross entropy.  
Алгоритм оптимизации - Adam со стандартными значениями параметров.  

Вероятно, данный вариант не является наиболее оптимальным, но показал достаточную точность.

Реализацию можно увидеть в [файле](https://github.com/podlinovvv/celeste_tracking/blob/master/net.py)

### Дальнейшие действия:

Был записан и разбит на кадры видеофайл с игровым процессом.

Далее каждый кадр последовательно разбивался на фрагменты 50x50 пикселей и оценивался нейросетью.  
На основе кадра создавался новый, где фрагменты с персонажем выделялись зелёной рамкой.  

Из обработанных кадров был собран новый видеофайл.

---
## Результат работы (youtube):

<a href="http://www.youtube.com/watch?feature=player_embedded&v=DGEk1-UyQUc
" target="_blank"><img src="http://img.youtube.com/vi/DGEk1-UyQUc/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="320" height="240" border="10" /></a>


#### В подавляющем большинстве кадров персонаж определён корректно.