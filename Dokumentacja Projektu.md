# Perceptron

Najprostszy i najbardziej podstawowy, pojedyńczy matematyczny model sztucznego neuronu. Neuron rozumiemy jako podstawowa jednosta systemu nerwowego, np. człowieka. Jest on używany do klasyfikacji binarnej.
Perceptron składa się z:
- wejścia
- wag
- stałej uczenia
- funkcji aktywacji
- wyjścia

Celem klasyfikacji używając perceptronu, jest znalezienie takich wag, aby wartości wyjściowe były odpowiednio takie same jak wartości oczekiwane.


### PA
PA - Perceptron Algorithm, podstawowy algorytm perceptronu.

### BUPA
BUPA - Batch Update Perceptron Algorithm, grupowo odświeżany algorytm perceptronu.

### Problem xor
W porównaniu do funkcji AND, której elementy są liniowo separowalne, z funkcją xor jest większy problem, ponieważ nie jest on liniowo separowalny. W związku z tym, podstawowe obliczenia perceptronu nie są wystarczające do rozwiązania tego problemu. 

Przykładowym rozwiązaniem tego problemu jest na przykład podniesienie zbióru o jeden wymiar wyżej, z zastosowaniem kernelu RBF (radialna funkcja bazowej RBF).

## Opis fragmentu kodów

### Przykładowo wygenerowane wykresy

# Sieć Hopfielda
Sieć neuronowa, która tak samo jak perceptron posiada neurony oraz ich połączenia.

### Tryb synchroniczny

### Tryb asynchroniczny

## Opis fragmentów kodu

### Przykładowo wygenerowane wykresy

# Algorytm Propagacji Wstecznej

Posiadając dużą sieć neuronową z wieloma warstwami, używając algorytmu propagacji wstecznej jesteśmy w stanie modyfikować wagi we wszystkich jej warstwach. Korzystając z tego algorytmu cofamy się do tyłu, warstwa po warstwie, tak jak to nazwa wskazuje, dochodząc do wybranej przez nas warstwy, aby zmienić jej wagę.

### Energia całkowita
Korzystając z energii całkowitej, wagi są aktualizowane po prezentacji wszystkich wektorów wejściowych.

### Energia cząstkowa
W przypadku korzystania z energii cząstkowej, wagi są aktualizowane po każdym pojedynczym prezentowanym wektorze wejściowym.

## Opis fragmentów kodu

### Przykładowo wygenerowane wykresy
