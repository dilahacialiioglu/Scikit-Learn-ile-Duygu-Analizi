# Scikit Learn ile Duygu Analizi
 IMDB sitesinde yapılan kullancı verilerinden oluşan veri seti ile bir İngilizce cümlenin olumlu mu olumsuz mu olduğunu tahmin etmeyi amaçlayan sistem oluşturulur.

Yapılan çalışma 3 parça şeklinde paylaşılmıştır. İlk parça olan ön işleme bölümü verinin hazırlık sürecidir. Burada kelime harici bütün karakterler çıkarılır, tüm karakterler küçük harfe çevrilir, etkisiz kelimeler (stopwords) çıkarılır ve kelimelerin kökleri bulunur. 

İkinci parçada ise Decision Tree, Random Forest, Logistic Regression ve Support Vector Machine algoritmaları hiper parametreler ile birlikte tek tek eğitilmiş ve en doğru sonucu veren model belirlenmiştir. 

| Algoritma Adları  	| Decision Tree Classifier 	| Random Forest Classifier 	| Logistic Regression 	| Support Vector Machine 	|
|-------------------	|--------------------------	|--------------------------	|---------------------	|------------------------	|
| Doğruluk Oranları 	| 72.44                    	| 84.43                    	| 88.74               	| 89.35                  	|

Yukarıdaki tabloda görüldüğü gibi SUpport Vector Machine algoritması 89.35 doğruluk oranı ile en iyi sonucu vermiştir.

Üçüncü parçada ise Support Vector Machine Algoritması ile tahmin yapacak olan sistem oluşturulmuştur.


IMDB sitesinden alınan veri seti Train, Test ve Valid veri seti olarak 3 parça halinde bulunmaktadır. Toplam 50000 veri içeren bu veri setleri repo içerisine yüklenmiştir. Ayrıca veri kümesinin ön işleme sonucundaki hali de yüklenmiştir. Algoritmalar üzerinde modellerin denendiği ve sistemin oluşturulduğu python dosyaları bu düzenlenmiş veri ile çalışmaktadır.
