Etiketleri olan veri setleriyle çalıştık. Modele ve eğitime hazırdılar fakat gerçekte projeler böyle ilerlemiyor. Gerçekte bir veri seti ile değil bir "problem" ile uğraşılır.

Makine öğrenmesinin evrensel işleyişi üç temel yapı üzerine kuruludur: 
1. **Görevi belirle**: Problemin ne olduğunu anla. Veri setini topla, verilerin nasıl ifade edildiğini anla, başarıyı nasıl ölçeceğini belirle. 
2. **Model geliştir**: Model evaluation protokollerini ve referans değerlerini (baseline) belirle. Genelleme becerisi için öncelikle eğit, sonrasında overfit yap ve istenilen sonuç gelene kadar regülerizasyon ve tune ederek geliştir. 
3. **Modeli yüklemek**: Model nerede çalışacaksa (mobil uygulama, web sitesi, gömülü sistem...) oraya yüklemek.

# Görevi belirlemek 
- Girişimiz ne ve neyi tahmin etmek istiyoruz? 
- Hangi makine öğrenmesi problemi ile karşı karşıyayız? 
	Binary classification, multiclass classification, scalar regression, vector regression, multiclass, multilabel classification image segmentation, ranking, clustering, generation, reinforcement learning... 
	Hangisinde ilerleyeceksek prosedürler ona göre değişiyor. 
- Mevcut çözüm ne? Makinenin yapacağı iş şu anda nasıl çözülüyor? Mevcut sistemi anlamak gerekir. 

Bir projeye başlarken iki hipotezimiz olmalı: 
1. Verilen girişlere karşın tahmin edilebilir çıkışlar olmalı.
2. Veriler ulaşılabilir olmalı ve giriş-çıkış arasında bir ilişki olmalı.


## Veri setini toplamak 
Düzgün ve işe yarar bir veri seti oluşturmak her şeyden önemli. Kısıtlı bir zaman varsa bu zamanın çok büyük bir kısmı daha büyük ve daha kaliteli bir veri seti oluşturmakla geçmeli. 

Önceki çalışmalarda etiketlenmiş veri setleriyle ilgilendik fakat çoğu zaman tüm bu verileri tek tek bizim etiketlememiz gerekebilir. 

Önemli olan kısım da verilerin çeşitliliği. Siz profesyonel kameralar ile çekilen, aynı restorandaki, aynı tabaklardaki fotoğraflarla modeli eğitirseniz kullanıcının farklı bir restoran farklı bir tabakla, farklı bir yemekle çektiği fotoğrafı tanımakta zorlanırız. Çünkü ortam karanlık olabilir, telefonun kamerası daha kötü olabilir. Her duruma ayak uydurabilecek şekilde eğitim yapılmalı. Buna `representative` denir.

Bunun dışında zaman parametresi de işin içine giriyor. Çok eski bir veri seti üzerinden şu anki problemlere çözüm bulmak, bir tahmin üretmek muhtemelen doğru olmayacaktır çünkü zamanla birçok şey değişmiş olacak. Buna da `concept drift` denir.

Veri ayrıca homojen olmalı, farklı kategorideki verilerin aşağı yukarı benzer örnek sayısına sahip olması bir kategorinin baskın olmaması için önemli. 

## Verileri anlamak 
- Fotoğraf, doğal dil metni gibi verilerimiz varsa birkaç veriye ve etiketine bakılabilir. 
- Sayısal feature'lar varsa bir histogram çizilebilir.
- Konum bilgisi varsa harita üzerine çizilebilir. 
- Bir sınıflandırma problemi varsa sınıfların dengeli olup olmadığına bakılabilir.
- `Target leaking`, modelin tahmin etmeye çalıştığı hedef hakkında doğrudan ipucu almasıdır. Bu, öğrenme sürecini bozar ve sahte başarıya neden olur.

## Başarı kıstası 
- Dengelenmiş bir sınıflandırma problemi için "accuracy" ve "ROC AUC" yaygın metriklerdir. 
	ROC AUC, modelin pozitif ve negatif sınıfları ayırt etme başarısını 0–1 arasında bir değerle ölçer.
	1 → Mükemmel ayırıyor, 0.5 → Rastgele tahmin gibi, 0 → Tamamen ters.
- Dengesiz sınıflar, sıralama problemleri veya multilabel sınıflandırma durumlarında; precision, recall ve ağırlıklı accuracy/ROC AUC tercih edilir. 


# Model geliştirmek 
En zor kısım problemi anlamak, verileri toplamak, etiketlemek ve temizlemek. Modeli geliştirmek ise en kolay kısmı. 

## Verileri hazırlamak 
Verileri girişe uygun hale getirmemiz gerekir. Genellikle derin öğrenme modelleri ham veriyi direkt olarak kullanamaz. `Vectorization`, `normalization`, `kayıp verilerle ilgilenmek` gibi yöntemler vardır. 

### Vectorization
Sinir ağları genel olarak float tensorları input olarak alır. Bu sebeple ses, görsel, metin gibi verileri öncelikle bir tensor'e çevirmemiz gerekir. Bu yönteme vectorization denir. 

Ev fiyatı tahmini veya rakamları sınıflandırma gibi sayısal veri setlerinde zaten girişler tensor yapısında olduğu için bu adım geçilebilir.

### Value normalization
Verilen feature'ların değerleri birbirinden çok farklı olabilir. Bazıları 0-100 arasında olurken bazıları 0-1 arasında olabilir ve bu dengesizliğe sebep olur. Bir feature'ın çarpanı 50 iken diğerinin 0.2 olması birini diğerinden çok daha baskın hale getirir. Bunları tek bir aralığa sabitlemek (örneğin 0-1 arasına) birbirlerine karşı dengeli hale getirir. 
	- Bunu yaparken de mean (ortalama)
	- standard deviation (standart sapma) kullanılır.
	
	
### Kayıp verilerle ilgilenme 
Bazı özellikler her örnekte bulunmaz. Bulnarın yerinin doldurulması ya da tamamen o özelliğin es geçilmesi gerekir. Kategori bazlı bir özellik ise "bu değer kayıp" diyebiliriz fakat sayısal bir özellik ise yerine 0 gibi bir rakam vermeyiz çünkü grafikteki sürekliliği bozar. 


## Evaluation protokolünün seçilmesi 
Amacımız modelin genelleme becerisi kazanması. Bunu da validation (geçerleme) metrikleri ile sağlıyouz. Farklı genelleme metotları görmüştük. 
	- Holdout validation set
	- K-fold cross-validation
	- itarated k-fold validation
	
## Baseline'ı geçmek 
İstatistiksel bir güç amaçlıyoruz. Amacımız modelin belli bir kapasiteyi, referansı aşabilmesi. Bunun bazı yöntemleri ise : 
	- Feature engineering
	- Doğru mimarinin seçimi: densely connected network, convnet, RNN, transformer? 
	- Yapılandırmaların doğruluğu ve uygunluğu: loss function, batch_size, learning rate
	
### Doğru loss fonksiyonunu seçmek için ufak bir yönlendirme

|Problem tipi | aktivasyon fonksiyonu | Loss fonksiyonu 
|---|---|---|
|Binary classification | sigmoid | binary_crossentropy|
|Multiclass, single-label classification|softmax|categorical_crossentropy|
|Multiclass, multi-label classification|sigmoid|binary_crossentropy|

## Modeli overfit etmek 
Model ne kadar iyi çalışıyor, kapasitesi ne kadar yüksek bunu test edebilmek için modeli overfit ederiz. Bir modeli nasıl overfit ederiz:
	1. Katman eklenebilir
	2. Katmanlar daha büyük olabilir
	3. Daha fazla epoch ile eğitilebilir
	
	
## Modeli regüle etmek ve tune 
Modelimiz artık istatistiksel bir güce erişti, overfit olabiliyor. Şimdi ince ayarlarını yapıp genellemeyi maksimum seviyeye çıkarabiliriz.  Peki ya nasıl: 
	- Farklı bir mimari denenebilir: katman ekle veya çıkar
	- Dropout ekle
	- Model küçükse, L1 ve L2 regularization eklenebilir
	- Farklı hiperparametreler (katmanlardaki unitler, optimizer'ın learning rate'i)
	- feature engineering
	
KerasTuner gibi yazılımlar ile hiperparametrelerin ince ayarı otomatik olarak yapılabilir. 

Model yapılandırması tamamlandıktan sonra, elimizdeki tüm eğitim verisi (training + validation) ile son üretim modeli eğitilir.
Bu modelin performansı son kez test verisiyle ölçülür.

### Test Sonucu Beklenenden Kötü Çıkarsa Ne Anlama Gelir?
Eğer test verisindeki performans, validation verisindeki performanstan önemli ölçüde düşükse, iki olasılık vardır:
1. Validation prosedürü hatalı olabilir:
	- Validation seti modeli değerlendirmek için yeterince temsil edici olmamış olabilir.
	- Örneğin veriler rastgele bölünmemiş ya da dağılım dengesiz olabilir.

2. Validation verisine overfit (aşırı uyum) olmuş olabilirsin:
	- Modelin hiperparametrelerini validation verisine göre ayarlarken, model bu veri setine fazla öğrenmiş olabilir.
	- Gerçek dünyada karşılaşılacak farklı örneklerde başarısız olur.
	
**ÇÖZÜM**: Daha güvenilir değerlendirme protokolleri kullanmak
K-Fold Cross-Validation gibi yöntemler önerilir:
	- Eğitim/validation ayrımı birden çok kez farklı şekillerde yapılır.
	- Performans daha sağlam ve genellenebilir biçimde ölçülür.


# Deploy the model 

Deploy.

 
