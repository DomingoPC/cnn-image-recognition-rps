# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:30:37 2024

@author: domin
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Calcular el tamaño (Mb) de n parámetros como se muestra en Colab
# Lo uso para tener una referencia más entendible del tamaño
# del modelo
def memoria_modelo(n_parametros, tipo):
    print(
        f'Memoria de parámetros {tipo}: {np.round(n_parametros * 4 / 1048576, 2)} Mb'        
        )
    

# Visualización de las activaciones de los filtros convolucionales
def visualizacion_filtros(model, n_top_layers=None, n_rows_grid=None, steps=50, clasificacion=''):
    # Para estudiar solo las capas convolucionales
    # descongeladas de los modelos con trasfer learning
    if n_top_layers:
        check_layers = model.layers[-n_top_layers:]
    else:
        check_layers = model.layers
        
    # Si se especifica el número de filas, no hará falta calcularlo de nuevo
    if n_rows_grid:
        n_filas_fijado = True
    
    # Mostrar filtros de las capas convolucionales
    for layer in check_layers:
        if 'conv' in layer.name:
        
            size = 128
            margin = 5
        
            # Colab AI:
            # Get the number of filters in the current layer
            num_filters = layer.output.shape[-1]
        
            # Número de filas del mallado de imágenes si tenemos 8 columnas
            if not n_filas_fijado:
                # Mostrar todos los filtros posibles
                n_rows_grid = int(np.ceil(num_filters / 8))
            
            # Imagen vacía que almacenará todas las activaciones de los filtros 
            # separadas por margenes
            results = np.zeros((n_rows_grid * size + (n_rows_grid - 1) * margin, # n filas
                                8 * size + (8 - 1) * margin, # n columnas
                                3)) # rgb
        
            results += 255
            
            for i in range(n_rows_grid):  # iterate over the rows of our results grid 
                for j in range(8):  # iterate over the columns of our results grid
        
                    filter_index = i + (j * 8)
                    # Break if we've processed all filters
                    if filter_index >= num_filters:
                        break
        
                    layer_output = model.get_layer(layer.name).output
        
                    # IA COlab:
                    loss_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x[:, :, :, filter_index]))
                    loss = loss_layer(layer_output)
        
                    loss_model = tf.keras.models.Model(inputs=model.inputs,
                                                       outputs=loss)
        
        
                    # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                    filter_img = generate_pattern(loss_model, size=size, steps=steps, lr=1)
                    filter_img = deprocess_image(filter_img[0].numpy())
        
                    # Put the result in the square `(i, j)` of the results grid
                    horizontal_start = i * size + i * margin
                    horizontal_end = horizontal_start + size
                    vertical_start = j * size + j * margin
                    vertical_end = vertical_start + size
                    results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img
        
            # Display the results grid
            height = int(n_rows_grid / 8)
            plt.figure(figsize=(20, np.where(height==0, 10, 20 * height))) # aspect ratio
            plt.title(layer.name, weight='bold')
            plt.imshow(results/255)
            plt.axis('off')
            plt.savefig(f"images/{clasificacion}/activaciones/{layer.name}.jpeg", bbox_inches='tight')
            plt.show()


# Estudio de salidas intermedias del modelo: generar patrones
def generate_pattern(loss_model, size=150, steps=40, return_steps=False, lr=1.):
    # Creamos una imagen aleatoria
    im = tf.constant(np.random.random((1, size, size, 3)) * 20 + 128.)

    # Ejecutamos el gradiente ascendente
    lr = 1.

    ax = None
    images = []
    loss = tf.constant(0.0)
    for _ in range(steps):

        with tf.GradientTape() as tape:
            tape.watch(im)
            loss = loss_model(im)

        # calculamos los gradientes y actualizamos
        grads = tape.gradient(loss, im)

        grads /= (tf.math.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

        im += grads * lr

        if return_steps:
            #arr = np.copy(im)
            images.append(im)


    if return_steps:
        images.append(im)
        return images
    else:
        return im

# Hemos de post-procesarlo para convertirlo 
# en valores enteros en el rango [0, 255] y poder mostrarlo
def deprocess_image(x):
    # normalizamos el tensor: centrado en 0. y con std 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # desplazamos 0.5 para hacerlos positivos y eliminamos lo que este fuera del rango [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convertimos a RGB
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Mostrar imágen superpuesta con su heatmap correspondiente
def show_images(*args, tags=None, title=None, clasificacion=''):
    n = len(args[0])
    plt.figure(figsize=(20, 4*len(args)))
    if title:
        plt.title(title, weight="bold")
        plt.axis('off')
        
    for i in range(n):
        for j in range(len(args)):
            data = args[j][i]
            img = data['image']
            ax = plt.subplot(len(args), n, i + 1 + j*n)
            cmap = None if 'cmap' not in data else data['cmap']
      
            if len(img.shape) == 3 and img.shape[2] == 1:
                ax.imshow(img.reshape(img.shape[:2]), cmap=cmap)
            else:
                ax.imshow(img, cmap=cmap)
      
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
      
            if 'tag' in data:
                tags = data['tag']
                n_lines = len(tags.split('\n'))
                ax.text(0.0,-0.1 * n_lines, tags, size=12,
                        ha="left", transform=ax.transAxes)        
    
    if title:
        plt.savefig(f"images/{clasificacion}/{title}.jpeg", bbox_inches='tight')
    
    plt.show()
    

# Mostrar heatmaps de las capas convolucionales
def show_heatmaps(images, model, alpha_heatmap=.4, n_finales=0, clasificacion=''):
    
    # Imágenes (raw) y tags por separado
    X = np.array([dic['image'] for dic in images])
    tags = [dic['tag'] for dic in images]
    
    # Mapas de calor de capas convolucionales
    for layer in model.layers[-n_finales:]:
        # print(layer)
        if 'conv' in layer.name.lower():
            # print("True")
            
            # Mapas de características: aplicar filtro a imagen
            get_feature_maps = tf.keras.models.Model(model.input,
                                                     model.get_layer(layer.name).output)
            feature_maps = get_feature_maps(X)
            
            # Usando mapas de características, damos heatmaps de esta capa
            heatmaps = []
            for idx in range(feature_maps.shape[0]):
                fm = feature_maps[idx]
                heatmap = np.sum(fm * get_feature_maps.get_weights()[-1][:], axis=-1)
                
                # Normalización y eliminación de valores <0                
                heatmap = np.clip(heatmap, 0, heatmap.max()) / heatmap.max()
                
                # Guardamos el heatmap
                heatmaps.append(heatmap)
                
            # Superponemos las imágenes con su heatmap correspondiente
            raw_images = np.copy(X) # toman valores en [0,1]
            
            superpuestas = []
            for im, hm, tag in zip(raw_images, heatmaps, tags):
                
                # Redimensión del heatmap al tamaño de la imagen
                heatmap = cv2.resize(hm, (im.shape[1], im.shape[0]))
                
                # Convertimos a RGB
                heatmap = 1 - np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Superposición ponderada
                imagen_superpuesta = heatmap * alpha_heatmap + (im*255) * (1-alpha_heatmap)
                
                superpuestas.append({'image': imagen_superpuesta / 255,
                                     'tag': tag})
                
            # Mostramos la capa convolucional y las imágenes superpuestas que genera
            show_images(superpuestas, title=layer.name, clasificacion=clasificacion)
        

# Mostrar imágenes test
def imagenes_test(model, test_seq, lista_idx, limite_imgs=10):
    count = 0
    clases_mapping = ['Piedra', 'Papel', 'Tijeras']
    images = []
    
    for idx in lista_idx:
        # Limitar número de imágenes
        count += 1
        if count > limite_imgs:
            break
        
        # Imágenes y textos
        im, clase_real = test_seq[idx]
        im = np.copy(im)
        
        pred = model.predict(im, verbose=0)
        texts = [f"\n\n-- Real: {clases_mapping[clase_real[0]]} --\n"
                 +'\n'.join(
            [f'{name}: {int(prob*100)}%' for (name, prob) in zip(clases_mapping, pred[0])]
        )]
        
        # [14] https://stackoverflow.com/questions/50630825/matplotlib-imshow-distorting-colors
        images.append(
            {'image': im[0][...,::-1], # rgb -> bgr para que imshow pinte colores originales
             'tag': texts[0]}
        )
        
    return images

    
