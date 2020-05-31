
from scipy.spatial import distance as dist #dizileri karşılastırmak ve islem yapmak için olan araclar
from imutils import perspective #google'nın API ile arayüz için gerekli olan bir mdoül
from imutils import contours #matplotlib ile çevre hesaplayan bir modul
import numpy as np
import argparse
import math
"""
argparse: argparse kullanıcıdan aldıgı parametreler için yardım mesajları, nasıl kullanıldıgına yonelik mesajlar üretir
"""
import imutils  #kolay görüntü işleme için kolaylık saglayan fonsksiyonları kapsayan modul. orn:rotation, traslation,resizing gibi...
import cv2

"""
öncelikle projeye baslamadan sunları bilmek gerekiyor:
    1)ölçüm yapacagamız referans cismimizin boyutunun bilinmesi gerekiyor
    2)ve ayrıca işleyeceğimiz görüntüdeki nesnelerin kolay saptanabilmesi gerekiyor
    3)ve son olarak referans objemiz her zaman resmin en solunda olmak zorundadır.
      algoritmamız objeleri bulurken en soldaki objeyi ref alacak sekilde tasarlandi.
"""
#verilen referans objenin ölcüsü ile biz resimdeki mesefaleri ölcebiliriz

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--resim", required=True,
	help="resmin path'i")
ap.add_argument("-w", "--kalinlik", type=float, required=True,
	help="en soldaki objenin gercek kalinligi")
args = vars(ap.parse_args())
"""
yukarida ki kodda, algoritmamızin calismasi için gerekli olan 2 inputu belirtiyoruz.
resim, işlem yapacagımız resmin path'i ve kalinlik ise referans nesnemizin genisligi. (cm cinsinden)
"""

# load the image, convert it to grayscale, and blur it slightly
resim = cv2.imread(args["resim"])
gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
gri_resim = cv2.GaussianBlur(gri_resim, (7, 7), 0)
"""
resmimizi yükleyip ardından tek kanala yani gri renge donustürüyoruz. son olarak da GaussianBlur fonsksiyonu ile bulanıklastırıyoruz.
bulanıklastirma isleminin bir diger adida low-pass filterdir. Low pass filter düşük frenkaslara izin veren, yüksek frekansların gecmesine
izin vermeyen bir filtredir. Frekans piksellerdeki değişim hızını ifade eder. resimde keskin kenarlarin yüksek frekansa sahiptir. blurlama
işlemi bu frekans gecislerinde ortalama alarak resmi yumusatır ve gürültüyü azaltır.
"""

koseler = cv2.Canny(gri_resim, 50, 100)
koseler = cv2.dilate(koseler, None, iterations=1)
koseler = cv2.erode(koseler, None, iterations=1)
"""
yukaridaki kodumuzun ilk satirinda canny fonsksiyonu ile görüntü üzerindeki kenarlarin tesbit ettik.
ardindan sirasi ile dilate ve erode fonsksiyonlarini cagirdik.
dilation: operatörü giriş olarak verilen görüntü üzerinde parametreler ile verilen alan sınırlarıni genisletmektir.
erosion: oper ise giriş olarak verilen görüntü üzerinde parametreler ile verilen alan sınırlarıni daraltmaktır.
"""


a_hat = cv2.findContours(koseler.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
a_hat = imutils.grab_contours(a_hat)
"""
öncelikle contour'u aciklamak gerekirse, resim üzerinde devam eden eğrideki tüm noktlari birlestirme olarak tanımlanabilir.
yukaridaki ilk olarak findContours fonsksiyonu ile nesnelerin contour(anahatlarını) belirliyoruz.
ardindan bu contourlari demet(tuple) veri tipinde döndürüyoruz.
"""
(a_hat, _) = contours.sort_contours(a_hat)
renk_paketi = ((255, 0, 0), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255)) #BGR degerleri
refObj = None
"""
belirledigimiz anahatları soldan saga dogru siraliyoruz.. buraya kadar gelmeden önce
bizim belirlediğimiz referans objemiz resimde en solda olmak zorunda oldugunu bilmemiz gerekiyor.
çünkü görüntü üzerinde contourlar belirlerken soldan-saga dogru gittigi için en soldaki nesnemizi
referans olarak alıyoruz. ve bu da a_hat listemizdeki ilk degerimiz oluyor.

daha sonra, refObj'mizden diğer nesnelere olan uzakligi belirtmek için dogrular çizecegiz.
bu dogruların renklerini belirlemek için renk listesi tanımlıyoruz.
"""

"""
ana_hat listemizdeki her contour için bir for dongüsü baslatıyoruz.
eğer contour değerimiz 100 den kucuk ise o nesneyi ignorla.
eğer contour degerimiz belirledigimiz(100)'den büyükse her nesne için nesneyi içine alan bir kutu çiziyoruz.
"""
for c in a_hat:
	if cv2.contourArea(c) < 100:
		continue

	kutu = cv2.minAreaRect(c)
	kutu = cv2.cv.BoxPoints(kutu) if imutils.is_cv2() else cv2.boxPoints(kutu)
	kutu = np.array(kutu, dtype="int")

	kutu = perspective.order_points(kutu)
    #order point ile kutularımızın sag-üst, sag-alt, sol-üst, sol-alt köşelerinin kordinatlarını yeniden ayarlıyoruz. bu noktalar uzaklik bulmada yardımci olacak
	cX = np.average(kutu[:, 0])
	cY = np.average(kutu[:, 1])
    #daha sonra numpy modülündeki average fonsksiyonu ile kutularımızın kordinatlarının ortalamasını alarak kutuların merkezini tesbit ediyoruz.

	if refObj is None:
	    (tl, tr, br, bl) = kutu
        #refObjmizin üstSol,üstSag, altSag,altSol koordinatlarını kutu demetinde tanımlıyoruz
		#eğer referans objemiz yok ise algoritmamızı baslatmamız gerekir..
	    (tlblX, tlblY) = midpoint(tl, bl)
	    (trbrX, trbrY) = midpoint(tr, br)
        #ilk olarak kutumuza TopLeft, TopRight, BottomLeft, BottomRight olarak 4 köşe koordinat belirliyoruz.
        #ardindan bu koselerin yazdıgımız orta nokta fonsksiyonu ile orta noktalarını buluyoruz
	    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        #SQRT((trbrx-tlblx)^2+(tbry-tlbly)^2) euclidean fonsksiyonu ile öklid uzunluklarını buluyoruz.
        refObj = (kutu, (cX, cY), D / args["kalinlik"])
        #burada refObj'miz 3 elemanli bir demet döndürüyor. ilki kutu, ikinisi refObj'nin merkezi(agırlık merkezi) ve son olarak uzunluklarını
        #bu uzunluk degerini soyle buluyoruz. görüntü üzerinden ölcütügümüz D öklid mesafesini, programın basında girecegimiz refObj'nin orjinal uzunluguna bölüyoruz.(cm)
        continue

        #burdan sonrasında resim üzerine çizdirme işlemleri yapacağız.
	orig = resim.copy()
	cv2.drawContours(orig, [kutu.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

    #ilk olarak refObj'mizle incelediğimiz objenin contourlarını cizdiriyoruz

	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([kutu, (cX, cY)])

    #vstack fonsksiyonu dizileri sütun matris şekilde yıgın yapmak için kullanılır.
    #buradan sonra sırası ile köselerin ve merkezlerin birbirlerine uzaklıgını ölcmeye hazırız.


	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, renk_paketi):
		#refObj ile ilgilendiğimiz nesneye karsılık gelen x-y kordinatları üzerinde bir döngü baslatıyoruz.
		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
			color, 2)

		#ardindan x-y noklatalarını temsil eden circllar cizdiriyoruz ve bu circllar arasında dogrular çizin onların uzunlugunu buluyoruz.

		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
        #daha sonra refObj ile ölçmek istedigimiz nesne arasındaki öklid mesafesini hesapliyoruz. yine burada buldugumuz degeri pixel->cm olarak verdigimiz degere bölerek gercek degeri buluruz.
        (mX, mY) = midpoint((xA, yA), (xB, yB))
		cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        #daha sonra buldugumuz degeri cizdigimiz dogrunun tam ortasına yazdırmak için orta noktalarını belirleyip belirledigimiz formatta yazdırıyoruz.

		cv2.imshow("resim", orig)
		cv2.waitKey(0)
        #ve resmin cıktısı. tüm elemanları ölcene dek bir tusa basilir.
