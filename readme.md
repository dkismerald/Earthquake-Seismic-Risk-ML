# Seismic Risk Prediction

Projekt realizujący analizę aktywności sejsmicznej z wykorzystaniem danych historycznych, przetwarzania danych w oknach czasowych oraz modelu uczenia maszynowego.

System analizuje aktywność sejsmiczną w wybranych regionach świata i na podstawie danych historycznych oraz aktualnych obserwacji wyznacza względne prawdopodobieństwo wystąpienia kolejnego zdarzenia sejsmicznego w określonym horyzoncie czasowym.

Projekt został przygotowany w ramach przedmiotu "Przetwarzanie danych strumieniowych".

---

## 1. Cel projektu

Głównym celem projektu jest opracowanie kompletnego pipeline'u do przetwarzania danych sejsmicznych, obejmującego:

1. pobranie i przygotowanie danych historycznych,
2. podział danych według regionów geograficznych,
3. utworzenie cech opisujących aktywność sejsmiczną,
4. przygotowanie zbioru treningowego,
5. wytrenowanie modelu uczenia maszynowego,
6. zapisanie wytrenowanego modelu,
7. pobranie aktualnych danych sejsmicznych z API,
8. utworzenie aktualnych cech dla poszczególnych regionów,
9. wykonanie predykcji,
10. przedstawienie rankingu regionów według oszacowanego ryzyka.

Projekt łączy klasyczne przetwarzanie danych czasowych z metodami uczenia maszynowego.

---

## 2. Dane do analizy

Do analizy został użyty dataset:

**All the Earthquakes Dataset : from 1990-2023**

Źródło:

Kaggle - All the Earthquakes Dataset : from 1990-2023

Dataset zawiera około trzech milionów rekordów dotyczących trzęsień ziemi zarejestrowanych na całym świecie w latach 1990-2023.

Każdy rekord reprezentuje pojedyncze zdarzenie sejsmiczne.

Wśród dostępnych informacji znajdują się między innymi:

- czas wystąpienia zdarzenia,
- lokalizacja geograficzna,
- szerokość geograficzna,
- długość geograficzna,
- magnituda,
- głębokość,
- region,
- typ danych,
- informacja o tsunami,
- znaczenie zdarzenia.

Dane historyczne są wykorzystywane do zbudowania zbioru treningowego dla modelu ML.

Źródło danych:

https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023

Dataset jest wykorzystywany jako źródło danych historycznych, natomiast aktualne dane podczas działania programu mogą być pobierane niezależnie za pośrednictwem API.

---

## 3. Ogólna architektura

Pipeline projektu można przedstawić następująco:

    Dane historyczne CSV
            |
            v
    Przygotowanie danych
            |
            v
    Podział na regiony
            |
            v
    Feature engineering
            |
            v
    Zbiór treningowy
            |
            v
    Model LightGBM
            |
            v
    Zapis modelu
            |
            |
            v
    -------------------------
    Aktualne dane z API
            |
            v
    Aktualne cechy regionów
            |
            v
    Wytrenowany model
            |
            v
    Predykcja ryzyka
            |
            v
    Ranking regionów


---

## 4. Organizacja projektu

Przykładowa struktura projektu:

    project/
    |
    +-- data/
    |   +-- earthquakes_1990_2023.csv
    |
    +-- features/
    |   +-- build_features.py
    |
    +-- predictor/
    |   +-- region_predictor.py
    |
    +-- streaming/
    |   +-- live_checker.py
    |
    +-- training/
    |   +-- train_ml_model.py
    |
    +-- model/
    |   +-- earthquake_model.joblib
    |
    +-- src/
    |   +-- utils.py
    |
    +-- config.py
    +-- main.py
    +-- requirements.txt
    +-- README.md
    +-- .gitignore

Poszczególne moduły odpowiadają za różne etapy przetwarzania danych.

---

## 5. Przygotowanie danych

Dane historyczne są wczytywane z pliku CSV.

Pierwszym etapem jest normalizacja danych, w szczególności:

- konwersja czasu do formatu datetime,
- ujednolicenie strefy czasowej UTC,
- sortowanie zdarzeń chronologicznie,
- ujednolicenie nazw kolumn,
- odfiltrowanie niepoprawnych rekordów.

Następnie dane są przypisywane do zdefiniowanych regionów geograficznych.

Każdy region posiada:

- nazwę,
- współrzędne środka,
- lokalizację wykorzystywaną przez API.

Dzięki temu możliwe jest niezależne analizowanie aktywności sejsmicznej dla wielu obszarów.

---

## 6. Feature engineering

Zamiast przekazywać pojedyncze rekordy bezpośrednio do modelu, z danych historycznych tworzone są cechy opisujące aktywność sejsmiczną w określonych oknach czasowych.

W projekcie wykorzystano między innymi następujące cechy:

### count_24h

Liczba zdarzeń sejsmicznych zarejestrowanych w ciągu ostatnich 24 godzin.

### count_72h

Liczba zdarzeń zarejestrowanych w ciągu ostatnich 72 godzin.

### max_mag_72h

Największa magnituda odnotowana w ciągu ostatnich 72 godzin.

### mean_mag_72h

Średnia magnituda zdarzeń z ostatnich 72 godzin.

### mean_depth_72h

Średnia głębokość zdarzeń z ostatnich 72 godzin.

### min_depth_72h

Najmniejsza głębokość zdarzenia z ostatnich 72 godzin.

### hours_since_last

Liczba godzin od ostatniego zarejestrowanego zdarzenia.

W efekcie pojedynczy stan regionu jest reprezentowany przez wektor:

    X = [
        count_24h,
        count_72h,
        max_mag_72h,
        mean_mag_72h,
        mean_depth_72h,
        min_depth_72h,
        hours_since_last
    ]

Takie podejście pozwala przekształcić dane zdarzeniowe w strukturę odpowiednią dla klasyfikatora ML.

---

## 7. Tworzenie etykiety

Model jest traktowany jako klasyfikator binarny.

Dla każdego punktu czasowego sprawdzane jest, czy w określonym przyszłym horyzoncie czasowym wystąpiło zdarzenie spełniające ustalone kryterium magnitudy.

W przypadku wystąpienia takiego zdarzenia:

    label = 1

W przeciwnym przypadku:

    label = 0

Model uczy się więc zależności pomiędzy aktywnością sejsmiczną obserwowaną przed danym momentem a wystąpieniem zdarzenia w późniejszym okresie.

---

## 8. Model uczenia maszynowego

Do klasyfikacji wykorzystano algorytm LightGBM.

LightGBM jest algorytmem opartym na gradient boosting i drzewach decyzyjnych.

Model otrzymuje zestaw cech opisujących aktywność danego regionu i zwraca prawdopodobieństwo należenia próbki do klasy pozytywnej.

Przykładowo:

    SanFrancisco_US -> P = 0.592

oznacza, że model przypisał analizowanemu stanowi regionu wartość 0.592 dla klasy pozytywnej.

Wartość ta jest wykorzystywana przede wszystkim do porównywania regionów między sobą.

---

## 9. X shape

Podczas działania programu wyświetlana jest informacja:

    X shape: (3, 7)

Oznacza ona:

- 3 - liczba analizowanych próbek,
- 7 - liczba cech wejściowych.

W tym przypadku analizowane są trzy regiony, a każdy z nich jest opisany siedmioma cechami.

Przykładowo:

    X shape: (3, 7)

można interpretować jako:

    Region 1 -> 7 cech
    Region 2 -> 7 cech
    Region 3 -> 7 cech

Jest to standardowy sposób przedstawiania wymiarów macierzy danych wejściowych w bibliotekach takich jak NumPy, pandas oraz scikit-learn.

---

## 10. Trenowanie modelu

Model jest trenowany na danych historycznych.

Proces treningowy obejmuje:

1. wczytanie datasetu,
2. przygotowanie danych czasowych,
3. utworzenie cech,
4. utworzenie etykiet,
5. podział danych na zbiór treningowy i walidacyjny,
6. trening modelu LightGBM,
7. obliczenie predykcji na zbiorze walidacyjnym,
8. obliczenie metryk,
9. zapis wytrenowanego modelu.

Model jest zapisywany na dysku za pomocą biblioteki joblib.

Dzięki temu nie ma konieczności ponownego trenowania modelu przy każdym uruchomieniu programu.

---

## 11. Walidacja modelu

Do oceny modelu wykorzystano między innymi:

- ROC AUC,
- Average Precision.

ROC AUC pozwala ocenić zdolność modelu do rozróżniania klas.

Average Precision jest szczególnie przydatną miarą w sytuacji, gdy klasy nie są rozłożone równomiernie.

Podczas treningu stosowany jest stały seed, dzięki czemu eksperyment można powtórzyć z zachowaniem tego samego podziału danych i porównywalnych wyników.

---

## 12. Przetwarzanie aktualnych danych

Po wytrenowaniu modelu nie jest on trenowany ponownie podczas każdego cyklu działania programu.

Program pobiera aktualne dane sejsmiczne z API, a następnie:

1. pobiera ostatnie zdarzenia dla regionu,
2. oblicza aktualne cechy,
3. tworzy macierz X,
4. przekazuje X do istniejącego modelu,
5. otrzymuje predykcję,
6. sortuje regiony według wartości predykcji.

Dzięki temu trening i predykcja są rozdzielone.

### Trening

    Dane historyczne
          |
          v
    Feature engineering
          |
          v
    ML training
          |
          v
    model.joblib

### Predykcja

    Aktualne dane API
          |
          v
    Feature engineering
          |
          v
    model.joblib
          |
          v
    Predykcja


---

## 13. Działanie programu

Program główny działa cyklicznie.

W każdym cyklu wykonywane są:

1. pobranie aktualnych danych,
2. przygotowanie cech,
3. predykcja,
4. sortowanie wyników,
5. wyświetlenie rankingu.

Przykładowy wynik:

    === SEISMIC RISK (ML MODEL) ===

    SanFrancisco_US      -> P=0.592
    HappyValley_AK       -> P=0.124
    LosAngeles_US        -> P=0.011

Najwyżej sklasyfikowany region znajduje się na początku listy.

Program może działać przez określony czas albo w sposób ciągły, dopóki użytkownik nie przerwie jego działania.

Program można zatrzymać za pomocą:

    Ctrl+C

---

## 14. Przykładowe cechy wejściowe

Przykładowy stan analizowanych regionów:

    SanFrancisco_US:
        count_24h = 80
        count_72h = 80
        max_mag_72h = 4.2
        mean_mag_72h = 2.25225
        mean_depth_72h = 7.363
        min_depth_72h = 1.17
        hours_since_last = 3.57

    HappyValley_AK:
        count_24h = 4
        count_72h = 4
        max_mag_72h = 3.6
        mean_mag_72h = 2.5
        mean_depth_72h = 102.7
        min_depth_72h = 65.7
        hours_since_last = 7.76

    LosAngeles_US:
        count_24h = 1
        count_72h = 1
        max_mag_72h = 1.38
        mean_mag_72h = 1.38
        mean_depth_72h = 2.19
        min_depth_72h = 2.19
        hours_since_last = 15.05


---

## 15. Przykładowy wynik predykcji

    === SEISMIC RISK (ML MODEL) ===

    SanFrancisco_US      -> P=0.592
    HappyValley_AK       -> P=0.124
    LosAngeles_US        -> P=0.011

W tym przypadku San Francisco zostało sklasyfikowane przez model jako region o najwyższym względnym ryzyku spośród analizowanych regionów.

Nie oznacza to, że trzęsienie ziemi z pewnością wystąpi.

Wartość P należy interpretować jako wynik modelu klasyfikacyjnego i miarę względnego ryzyka wynikającą z cech dostarczonych do modelu.

---

## 16. Wizualizacja

W projekcie wykorzystano również wizualizację zależności pomiędzy maksymalną magnitudą w ostatnich 72 godzinach a estymowanym ryzykiem.

Przykładowa obserwowana zależność pokazuje wzrost wartości predykcji wraz ze wzrostem maksymalnej magnitudy.

Wizualizacja pozwala łatwiej interpretować działanie modelu i przedstawiać zależności występujące w danych w formie graficznej.

---

## 17. Konfiguracja

Najważniejsze parametry projektu znajdują się w pliku:

    config.py

Można w nim określić między innymi:

- listę analizowanych regionów,
- współrzędne regionów,
- parametry okien czasowych,
- minimalną magnitudę,
- horyzont predykcji,
- seed,
- lokalizację zapisanego modelu,
- ustawienia GPU.

Dzięki centralnej konfiguracji zmiana parametrów projektu nie wymaga modyfikowania wielu modułów.

---

## 18. Uruchomienie

### Instalacja zależności

Po utworzeniu środowiska wirtualnego należy zainstalować wymagane biblioteki:

    pip install -r requirements.txt

### Trenowanie modelu

Trening wykonuje się poleceniem:

    python training/train_ml_model.py

Po zakończeniu treningu model zostanie zapisany w katalogu model/.

### Uruchomienie programu

Program główny uruchamia się:

    python main.py

Jeżeli projekt wykorzystuje parametry trybu pracy:

    python main.py --mode api --poll-interval 20 --duration 300

Parametr `--poll-interval` określa odstęp pomiędzy kolejnymi aktualizacjami.

Parametr `--duration` określa czas działania programu.

---

## 19. Obsługa błędów

Projekt został wyposażony w podstawową obsługę błędów związanych między innymi z:

- brakiem danych,
- błędami połączenia z API,
- problemami z odczytem plików,
- niepoprawnym formatem danych,
- niedostępnością GPU.

Jeżeli GPU nie jest dostępne lub konfiguracja GPU nie działa poprawnie, trening może zostać wykonany na CPU.

Dzięki temu działanie projektu nie jest uzależnione od obecności konkretnego sprzętu.

---

## 20. Powtarzalność eksperymentu

W projekcie zastosowano stałą wartość `SEED`.

Pozwala to ograniczyć wpływ losowości podczas przygotowywania danych i treningu modelu.

Dzięki temu możliwe jest ponowne przeprowadzenie eksperymentu w warunkach możliwie zbliżonych do poprzedniego uruchomienia.

---

## 21. Ograniczenia projektu

Należy podkreślić, że projekt nie stanowi systemu sejsmologicznego przeznaczonego do wydawania ostrzeżeń o rzeczywistych trzęsieniach ziemi.

Model analizuje statystyczne zależności występujące w danych historycznych.

Na wynik wpływają między innymi:

- jakość danych historycznych,
- sposób definiowania regionów,
- długość okien czasowych,
- wybrane cechy,
- sposób definiowania etykiety,
- nierównomierna liczba zdarzeń pomiędzy regionami,
- ograniczona liczba cech wejściowych.

Dlatego wartości predykcji należy traktować jako wynik modelu eksperymentalnego, a nie jako pewną prognozę przyszłego trzęsienia ziemi.

Głównym celem projektu jest pokazanie procesu przetwarzania danych zdarzeniowych, budowy cech czasowych i zastosowania modelu ML do analizy takich danych.

---

## 22. Dlaczego zastosowano podejście ML

Pierwsza część projektu polegała na klasycznej agregacji danych historycznych.

Takie podejście pozwalało obliczyć między innymi częstotliwość występowania zdarzeń w określonym regionie, jednak nie wykorzystywało w pełni możliwości uczenia maszynowego.

Dlatego zastosowano model klasyfikacyjny.

Model nie otrzymuje pojedynczego trzęsienia jako wejścia, lecz opis aktualnej aktywności sejsmicznej:

    liczba zdarzeń
    +
    magnitudy
    +
    głębokości
    +
    czas od ostatniego zdarzenia

Na tej podstawie model uczy się zależności pomiędzy stanem aktywności sejsmicznej a wystąpieniem zdarzenia w przyszłym oknie czasowym.

Jest to istotna część projektu, ponieważ pozwala przejść od prostego przetwarzania danych do rzeczywistego zastosowania algorytmu uczenia maszynowego.

---

## 23. Podsumowanie

Projekt przedstawia kompletny proces analizy danych sejsmicznych:

    dane historyczne
          |
          v
    czyszczenie danych
          |
          v
    analiza czasowa
          |
          v
    feature engineering
          |
          v
    dataset ML
          |
          v
    trening LightGBM
          |
          v
    walidacja
          |
          v
    zapis modelu
          |
          v
    aktualne dane
          |
          v
    predykcja
          |
          v
    ranking regionów

Najważniejszym elementem projektu jest połączenie przetwarzania danych czasowych z modelem uczenia maszynowego oraz wykorzystanie tych samych cech zarówno podczas treningu, jak i podczas późniejszej predykcji.

Projekt pozwala prześledzić cały cykl przetwarzania danych: od surowego zbioru historycznego, przez przygotowanie cech i trening modelu, aż do wykorzystania modelu na aktualnych danych.