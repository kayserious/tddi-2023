
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


Denenmiş diğer yöntemler ile kıyaslandığında ise;

|Dil Modeli                                                                    | CV|Epoch | Ortalama F Makro Skoru|
|:-----------------------------------------------------------------------------|--:|:-----|----------------------:|
|BERTurk Cased (simpletransformers) Domain Spesifik 3 epoch\w 32k size corpus  |  5|12    |                 0.9584|
|BERTurk Cased (simpletransformers) Domain Spesifik 3 epoch \w 18k size corpus |  5|6     |                 0.9437|
|BERTurk Cased (simpletransformers) Domain Spesifik 2 epoch \w 18k size corpus |  5|6     |                 0.9432|
|BERTurk Cased (simpletransformers)                                            |  5|10    |                 0.9430|
|BERTurk Cased (simpletransformers) Domain Spesifik 1 epoch \w 18k size corpus |  5|6     |                 0.9427|
|BERTurk Cased (simpletransformers) Domain Spesifik 4 epoch \w 18k size corpus |  5|6     |                 0.9411|
|BERTurk 128k Uncased (Domain Spesifik Denemesi \w 15k size corpus)            |  5|5     |                 0.9355|
|BERTurk Cased                                                                 |  5|10    |                 0.9346|
|BERTurk Cased (Domain Spesifik Denemesi & MLM  \w 4k size corpus)             |  5|5     |                 0.9333|
|BERTurk Cased                                                                 |  5|15    |                 0.9319|
|BERTurk Uncased                                                               |  5|10    |                 0.9301|
|BERTurk Cased (Domain Spesifik \w 2k size corpus)                             |  5|10    |                 0.9278|
|Distilled BERTurk                                                             |  5|8     |                 0.9109|
|TFIDF - BERTurk Embedding Catboost \w zemberek normalization                  |  5|-     |                 0.8961|
|TFIDF - BERTurk Cased Embedding Catboost                                      |  5|-     |                 0.8912|
|GPT-2                                                                         |  5|10    |                 0.8799|
|BERTurk Cased Embedding Catboost                                              |  5|-     |                 0.8711|
|BERTurk Uncased (simpletransformers)                                          |  5|6     |                 0.8581|


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



---------------


## Kaynaklar

Pretrain corpus için kullanılan kaynaklar;


[Toraman, C., Şahinuç, F., & Yilmaz, E. (2022). Large-Scale Hate Speech Detection with Cross-Domain Transfer. In Proceedings of the Language Resources and Evaluation Conference (pp. 2215–2225). European Language Resources Association.](https://aclanthology.org/2022.lrec-1.238/)

[Türkçe Sosyal Medya Paylaşımı Veri Seti @ Kaggle](https://www.kaggle.com/datasets/mrtbeyz/trke-sosyal-medya-paylam-veri-seti)