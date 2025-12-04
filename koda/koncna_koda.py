import numpy as np #Uvozimo Numpy za matrike in računanje lastnih vrednosti
import matplotlib.pyplot as plt #za risanje grafov
import matplotlib.colors as mcolors # za barvne mape

# naredimo svoj colormap, ki bo imel ozadje črno
turbo = plt.get_cmap("turbo") #vzamemo turbo iz colormapa
barve = turbo(np.linspace(0, 1, 256)) #vzamemo 256 barv enakomerno iz colormapa turbo
barve[0] = np.array([0, 0, 0, 1]) # zamenjamo prvo barvo iz colormapa s črno tako kot smo želeli
black_turbo = mcolors.ListedColormap(barve)

DPI = 150 #resolucija slike
FIGSIZE = (80, 80) #velikost slike v inčih, da bomo dobili 12000x12000

ST_MATRIK = 500000 #število matriki, ki jih bomo generirali
MIN_DIM = 8 #minimalna dimenzija matrike
MAX_DIM = 14 #maksimalna dimenzija matrike

coeff = np.array([-1, 0, 1]) #koeficienti izmed katerih bomo izbirali elemente matrike
GRID_POINTS = 4500 #resolucija histograma
RADIJ = 5 #območje izrisa v kompleksni ravnini

#Funkcija za generiranje lastnih vrednosti matrik
def generiraj_laste_vrednosti():
    rng = np.random.default_rng() #ustvarimo generator
    vrednosti = [] #sem shranimo lastne vrednosti

    #generiranje matrike
    for _ in range(ST_MATRIK):
        n = rng.integers(MIN_DIM, MAX_DIM + 1) #izberemo naključno dimenzijo

        #določimo diagonalo, superdiagonalo in subdiagonalo kot pri torplitzovih matrikah
        glavna_diag = rng.choice(coeff)
        super_diag = rng.choice(coeff)
        sub_diag = rng.choice(coeff)

        M = np.zeros((n,n)) #matrika samih ničel

        #dodamo diagonale od zgoraj
        M[np.arange(n), np.arange(n)] = glavna_diag
        M[np.arange(n - 1), np.arange(1, n)] = super_diag
        M[np.arange(1, n), np.arange(n - 1)] = sub_diag

        #za bolj zanimivo sliko dodamo še elemente na 2 in 3 subdiagonalo
        for k in range(2, 4):
            for j in range(n - k):
                M[j + k, j] = rng.choice(coeff)

        lastne_vrednosti = np.linalg.eigvals(M) 
        vrednosti.append(lastne_vrednosti)

    z = np.concatenate(vrednosti) #zdrušimo od vseh matrik lastne vrednosti v en sam vektor

    return z

def transformacije(z):
    z_nove = [] 

    # 10-člena simetrija
    for k in range(10):
        kot = 2*np.pi*k/10 #zavrtimo kompleksno število z za ta kot
        z_nove.append(z * np.exp(1j*kot))

    z_nove = np.concatenate(z_nove)

    #radialno popačenje oziroma warp
    #najprej jo pretvorimo v polarni zapis z_nove = r*exp(i * t)
    r = np.abs(z_nove) #radialna razdalja
    t = np.angle(z_nove) #da nam kot

    warp = 1 + 0.28*np.sin(8*t) + 0.12*np.sin(16*t) #tu sin(8t) doda 8-listno simetrijo, sin(16t) pa doda 16 majhnih valov.
    r2 = r*warp

    #Spirala, dodamo majhno odvisnost kota od radija, tako se točke rahlo zasukajo glede na razdaljo od izhodišča
    t2 = t + 0.07*np.sin(6*r)

    z2 = r2 * np.exp(1j*t2) #damo iz polarnega zapisa nazaj v standardni zapis

    return z2

def gostota(z):
    x = z.real
    y = z.imag

    robovi = np.linspace(-RADIJ, RADIJ, GRID_POINTS + 1) #robovi mreže histograma
    h, xe, ye = np.histogram2d(x, y, bins = [robovi, robovi]) #histogram gostote

    h = h.astype(float) #damo v float zaradi operacij

    #dodamo log transformacijo, da poudarimo tudi zelo nizke vrednosti
    h = np.log1p(h * 200)

    #normaliziramo
    h = h / (np.percentile(h, 99.9) + 1e-12)

    #odstranimo vrednosti izven željenega razpona
    h = np.clip(h, 0, 1)

    #dvignemo kontrast nizkih vrednosti z gamma korekcijo
    h = h **0.5

    return h, xe, ye

def main():
    print("Generiranje lastnih vrednosti")
    z = generiraj_laste_vrednosti()

    print("Dodaja transformacij")
    z2 = transformacije(z)

    print("Histogram")
    h, xe, ye = gostota(z2)

    print("Risanje grafa")
    fig = plt.figure(figsize=FIGSIZE, dpi = DPI, facecolor = "black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")

    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    ax.imshow(
        h.T,
        origin = "lower",
        extent = extent,
        cmap = black_turbo,
        interpolation= "bicubic",
        vmin = 0.0,
        vmax = 1.0
    )

    ax.set_axis_off()
    plt.tight_layout(pad = 0)

    fig.savefig("koncna_slika2.png", dpi = DPI, facecolor = "black", bbox_inches = "tight")
    plt.close(fig)

    print("Slika shranjena kot konacna_slika2.png")


if __name__ == "__main__":
    main()

