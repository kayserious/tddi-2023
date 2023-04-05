
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

Bahsettiğimiz Continual Pretrain yönteminden önce 