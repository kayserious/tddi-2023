
# TDDI 2023 - Kayserious Çözümü

Bu github dizini TEKNOFEST 2023 - Türkçe Doğal Dil İşleme yarışması için Kayserious takımının çözümünü içermektedir.

## Amaç

Çalışmanın amacı bir kişiye, gruba, ideolojiye yönelik saldırganca ya da saldırganca olmayan metinleri sınıflandırmaktır. 

Saldırganca bulunan metinler;

- Cinsiyetçi söylem
- Irkçı söylem
- Küfür
- Hakaret söylemleri

olarak etiketlenecektir.

## Yöntem

Uygun görünen ve yüksek doğrulama skorları elde edilebilecek BERT modellerini doğrudan kullanıp hazır hale getirmek yerine daha genelleyebilen ve benzer amaçlar için de tekrar tekrar kullanılabilecek bir model elde etmek adına "Öğrenmeye Devam Etme" (Continual Pretrain) yöntemini uygun gördüğümüz önceden eğitilmiş BERT modeline uyguladık.

Continual Pretrain yöntemi kısaca şöyle özetlenebilir;

Çeşitli Türkçe metinler kullanılarak önceden ağırlıkları belirlenmiş BERT modelinin ağırlıklarını bir sosyal medya derlemi üzerinden tekrardan denetimsiz olarak belirleme işlemidir.

Bu yöntem ile üstü kapalı ifadeler, sosyal medyaya has kullanımlar gibi BERT'in aslında bilmediği dilsel özellikleri de modelimize kazandırmış oluyoruz.

Bu işlem sırasında kullanılan Türkçe metinlerin kaynakları sayfa sonunda yer alacaktır.

## Mimari

![alt text](https://github.com/kayserious/tddi-2023/blob/main/metawork/scheme.png?raw=true)

Bahsettiğimiz Continual Pretrain yönteminden önce sınıflandırıcı oluşturmak için kullanılacak yol 1a - 1c şeklinde olacak iken sosyal medya verisi ile 1b adımını araya ekliyoruz. Bu kısmı `pretrainer.py` dosyasında bulabilirsiniz.

## Kullanım

Bütün bağımlılıkları içeren `kayserious_tddi.yaml` dosyası ile yeni bir sanal çevre oluşturarak çalışmayı test edebilirsiniz. 

```bash
conda env create -f kayserious_tddi.yaml
conda activate kayserious_tddi
```

### Uçtan uca yeniden çalıştırma

Çalışma sonucunu Windows ortamında sıfırdan tekrar oluşturmak adına öncelikle `base_model` klasörü altındaki `clone_models_from_hub.bat` isimli git kodları içeren dosya çalıştırılır ve bulunduğu klasöre ihtiyaç duyulan BERT modelinin en güncel sürümünü indirir.

Sonrasında tüm çalışma izleyen kod ile yeniden üretilebilir.

```bash
python run.py
```

### Tahminleme

`KAYSERIOUSModel` objesinin `live=` argümanı `True` olarak işaretlendiğinde eğitim modundan üretim moduna geçiş yapılabilir ve işlem `predictor.py` dosyasında fonksiyonel hale getirilmiştir.

```bash
python predictor.py --input_data teknofest_train_final.csv 
```

Yukarıdaki komut `--input_data` argümanıyla aldığı yoldaki veri ile tahminlemeyi çalıştırır ve `--output_data` argümanı boş geçildiyse çalıştırıldığı dizine `predictions.csv` ismiyle tahminleri kaydeder.