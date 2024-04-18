# Preguntas semilla 

## Contextualización
 **1. Número de distritos que hay en Madrid Capital**
- Son 21 distritos
```
'Arganzuela',
 'Barajas',
 'Carabanchel',
 'Centro',
 'Chamartín',
 'Chamberí',
 'Ciudad Lineal',
 'Fuencarral - El Pardo',
 'Hortaleza',
 'Latina',
 'Moncloa - Aravaca',
 'Moratalaz',
 'Puente de Vallecas',
 'Retiro',
 'Salamanca',
 'San Blas - Canillejas',
 'Tetuán',
 'Usera',
 'Vicálvaro',
 'Villa de Vallecas',
 'Villaverde'
```

 **2. Los 5 distritos con mayor precio medio por m2**
 ```
Salamanca    7374.0
Chamberí     6347.0
Centro       6144.0
Chamartín    6039.0
Retiro       5447.0
 ```
 **3. Los 5 distritos con menor precio por m2**
 ```
Villaverde            1988.0
Puente de Vallecas    2344.0
Usera                 2368.0
Carabanchel           2612.0
Latina                2699.0
 ```
 **4. Número de barrios en total**
 - Son 128
 
 **5. Número de barrios que hay en cada distrito**
 ```
 distrito
Ciudad Lineal            9
San Blas - Canillejas    8
Fuencarral - El Pardo    8
Arganzuela               7
Usera                    7
Latina                   7
Moncloa - Aravaca        7
Carabanchel              7
Chamberí                 6
Hortaleza                6
Chamartín                6
Moratalaz                6
Puente de Vallecas       6
Retiro                   6
Salamanca                6
Centro                   6
Tetuán                   6
Barajas                  5
Villaverde               5
Vicálvaro                2
Villa de Vallecas        2
 ```
 **6. Los 5 barrios con mayor precio medio por m2**
 ```
 neighbourhood_cleansed
Castellana          7374.0
Fuente del Berro    7374.0
Goya                7374.0
Guindalera          7374.0
Lista               7374.0
 ```
 **7. Los 5 barrios con menor precio por m2**
 ```
 neighbourhood_cleansed
Butarque         1988.0
Los Angeles      1988.0
Los Rosales      1988.0
San Andrés       1988.0
San Cristobal    1988.0
```

## Oferta
 **1. Oferta total de alquiler**
 - 18906 inmuebles (según esta base de datos)

 **2. Tipo de alquiler**
 - Son 4
 ```
 'Private room',
 'Entire home/apt',
 'Shared room',
 'Hotel room'
 ```
 **3. Oferta por tipo de alquiler**
 ```
 room_type
Entire home/apt    58.700942
Private room       39.088120
Shared room         1.364646
Hotel room          0.846292
 ```

## Distribución de la oferta
 **1. Los 5 distritos con mayor oferta de alquiler**
 ```
 distrito
Centro                   8432
Salamanca                1266
Chamberí                 1148
Arganzuela               1070
Tetuán                    811
 ```
 **2. Los 5 distritos con menor oferta de alquiler**
 ```
Villaverde                186
Barajas                   147
Moratalaz                 131
Villa de Vallecas         105
Vicálvaro                  71
 ```
**3. Los 5 barrios con mayor oferta de alquiler (y por distrito)**
 ```
 distrito               neighbourhood_cleansed
Centro                 Embajadores               2255
                       Universidad               1772
                       Palacio                   1533
                       Sol                       1125
                       Justicia                   910
 ```

**4. Los 5 barrios con menor oferta de alquiler**
 ```
 distrito               neighbourhood_cleansed
Fuencarral - El Pardo  Fuentelareina                5
Moncloa - Aravaca      Valdemarín                   4
Fuencarral - El Pardo  El Pardo                     3
Moratalaz              Horcajo                      2
Ciudad Lineal          Atalaya                      1
 ```

## Propietarios (en manos de quien está)
 **1. Distribución de las propiedades en propietarios**
 ![alt text](image.png)

- El 75% aprox tienen 7 o menos de 7 propiedades anunciadas
```
calculated_host_listings_count (# frecuencia relativa)
1      45.51
2      12.86
3       6.82
4       4.27
5       2.94
6       2.38
7       1.85


calculated_host_listings_count (# La frecuencia acumulada)
1       45.51 
2       58.37 
3       65.20
4       69.47
5       72.41
6       74.79
7       76.64
```
- Si bien existen una serie de grupos que poseen o gestionan, en número absoluto, un elevado número de apartamentos. Se ha identificado a algunos de ellos:
    - MitHouse (https://www.mithouse.es/)
    - Ivory Homes (https://www.ivoryhomes.es/)
    - Welcomer group (https://www.welcomergroup.com/es/)
    - Alter Home (https://alterhome.com/en)
    - Luxury Rental Madrid (https://luxuryrentalsmadrid.com/)


 **2. Tipos de alquiler que ofrecen**  
 (Esta variable se discretizará más adelante)
 - 1 propiedad 
```
 room_type
Entire home/apt    4638 (53.9%) #Porcentajes en relación a esta segmentación
Private room       3868 (44.95%)
Shared room          85 (0.98%)
Hotel room           13 (0.15%)
```
- Entre 2 y 5 propiedades:
```
room_type
Entire home/apt    2632 (51.7%) #Porcentajes en relación a esta segmentación
Private room       2356 (46.33%)
Shared room          61 (1.19%)
Hotel room           36 (0.70%)

```
- Entre 6 y 14 propiedades:
```
room_type
Entire home/apt    1492 (67.2%) #Porcentajes en relación a esta segmentación
Private room        585 (26.36%)
Shared room          84 (3.78%)
Hotel room           58 (2.6%)
```

- Más de 15 propiedades:
```
room_type
Entire home/apt    2336 (77.9%) #Porcentajes en relación a esta segmentación
Private room        581 (19.37%)
Hotel room           53 (1.7%)
Shared room          28 (0.93%)
```
Parece ser que el mercado del alquiler de apartamentos está copado por pequeños, medianos y grandes inversores


 **3. Distribución por propiedades en los Distritos**
- **Entire home/apt**  
Los 5 distritos con mayor número de apartamentos en alquiler son:
```
distrito
Centro        5994
Salamanca      880
Chamberí       644
Arganzuela     577
Tetuán         478
```
Los 5 distritos con menor número de apartamentos en alquiler son:
```
Barajas              42
Villaverde           42
Villa de Vallecas    28
Moratalaz            26
Vicálvaro            13
```
- **Private room**  
Los 5 distritos con mayor número de oferta de habitaciones privadas son:
```
distrito
Centro           2181
Arganzuela        481
Chamberí          475
Carabanchel       413
Ciudad Lineal     380
```
Los 5 distritos con menor oferta de habitaciones privadas en alquiler son:
```
Villaverde           131
Barajas              103
Moratalaz            103
Villa de Vallecas     74
Vicálvaro             57
```
- **Shared room**  
Los 5 distritos con mayor oferta de habitaciones compartidas son:
```
distrito
Centro           131
Chamberí          22
Villaverde        12
Ciudad Lineal     11
Latina            11
```
Los 5 distritos con menor oferta de habitaciones compartidas son:
```
distrito
Barajas                  2
Fuencarral - El Pardo    1
Hortaleza                1
Moratalaz                1
Vicálvaro                1
```
- **Hotel room**  
Los 5 distritos con mayor oferta de habitaciones de hotel son:
```
distrito
Centro       126
Salamanca     11
Chamberí       7
Hortaleza      5
Chamartín      4
```

Los 5 distritos con menor oferta de habitaciones de hotel son:
```
distrito
Ciudad Lineal         1
Moncloa - Aravaca     1
Moratalaz             1
Puente de Vallecas    1
Villaverde            1
```


## Tipo de vivienda
 
 1. Media de Nº de habitaciones
 2. Nº de baños medio
 3. Nº de camas medio
 4. Nº de potenciales huéspedes
 5. Superficie media de vivienda

## Precios de alquiler
 1. Precio medio de alquiler por tipo de oferta 
 2. Precio medio de alquiler por superficie de vivienda
 3. Precio medio de alquiler en base al número de huéspedes que se pueden alojar.
 4. Precio medio de alquiler por distrito en función del tipo de oferta
 5. Precio medio de alquiler por tipo de propietario

## Reviews y valoraciones
 1. Media de puntuación y valoraciones
 2. Media de puntuación por tipo de propietario
 3. Porcentaje de reviews por tipo de propietario
 4. Porcentaje de reviews por encima o igual a 4, y por debajo de 4, por tipo de propietario
 5. Porcentaje de reviews según la segmentación del mercado
 6. Nº de reviews y media de valoración por característica vivienda
 7. Nº de reviews y media de valoración por distrito
 8. Nº de reviews y media de valoración si es o no super_host y por tipo de propietario


## Relaciones entre otros factores
 31. Precio de alquiler medio vs ser super_host o no
 32. Precio de alquiler medio vs mayor números de reviews
 33. Precio de alquiler medio vs mayor tipo de valoración global
 34. Precios de alquiler vs proximidad a puntos de interés:
- Representaren el mapa con un color los puntos de interés  
- Ubicación de los inmuebles en relación a la distancia media de todos\
- Posicionar la ubicación de los "x" con menor media de distancia a todos.

## Ponderación de mejor oportunidad
Puedo establecer un prototipo ideal de piso con los valores del análisis anterior :
- Precio menor de compra (va implícito la superficie, y por ende el nº de habitaciones y el precio_m2).
- Mejor ubicado en cuanto a media a los diferentes puntos de interés
- Con mayor precio_medio de alquiler (podría hacerlo por superficie, es decir, una búsqueda para 50m2 y otra para 70m2)
- Mejor puntación en valoraciones
Y una vez lo tenga, mostar las características y su ubicación en el mapa.