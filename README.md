# ANALIZA MEDICINSKIH SLIK 2019-2020
# SEMINARSKA NALOGA: Zaznavanje sprememb lezij v MR slikah glave z nevronskimi mrežami

POPIS DATOTEK:
- AMS/README.md
- AMS/Python_paketi.txt # Beležka s seznamom nameščenih Python paketov in številkami njihovih različic.
- AMS/Seminar_UNET_abs.ipynb # Jupyter beležka z rešitvijo problema po prvi strategiji
- AMS/Seminar_UNET_rel.ipynb # Jupyter beležka z rešitvijo problema po drugi strategiji
- AMS/amslib_seminar.py # Python modul s funkcijami za nalaganje in pripravo podatkov
- AMS/models/ams-unet1-1-192-192-1-f16-j.h5 # shranjen model, pridobljen z beležko Seminar_UNET_abs.ipynb
- AMS/models/ams-unet1-1-192-192-2-f16-j.h5 # shranjen model, pridobljen z beležko Seminar_UNET_rel.ipynb
- AMS/data/README.txt # Opis podatkov (zaradi velikosti sama zbirka ni bila naložena na repozitorij.

OPIS ZGRADBE BELEŽK Seminar_UNET_abs.ipynb IN Seminar_UNET_rel-ipynb:

1. INICIALIZACIJA
- Nalaganje knjižnic (vključno z lastno knjižnico amslib_seminar.py)
- Inicializacija parametrov.
OPOMBA: Vsi nastavljivi parametri se nahajajo v tej celici (z izjemo parametra za uteževanje kategorij BETA, ki je definiran znotraj kriterijske funkcije weighted_crossentropy().

2. PRIPRAVA PODATKOV
- Nalaganje in priprava izhodnih podatkov.
- Nalaganje in priprava vhodnih podatkov.
- Razdelitev podatkov na učne in testne.

3. NAČRTOVANJE IN UČENJE MODELA
- Definicija novih modelov v obliki funkcij.
- Izgradnja novega modela s klicem izbrane funkcije.
- Prevajanje modela v strojno kodo s funkcijo compile().
- Učenje modela s funkcijo fit() in shranjevanje modela.

4. VREDNOTENJE MODELA
- Vrednotenje razgradnje (Diceov koeficient, razlika volumnov)
- Vrednotenje razvrščanja (točnost, občutljivost, specifičnost, ROC AUC)
