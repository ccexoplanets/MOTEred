# MOTEred
Scripts para reducir y extraer fotometría del proyecto MOTE:

Para instalar copie la carpeta scripts en alguna carpeta de su elección.
Una carpeta con imágenes debe existir al mismo nivel que la carpeta scripts.

El orden de ejecución es el siguiente:

1) Crear carpeta imagenes (al mismo nivel que carpeta scripts).
2) Dentro de imagenes crear la siguiente estructura:

   imagenes/INSTRUMENTO/YYYY-MM-DD/raw/bias
   
   imagenes/INSTRUMENTO/YYYY-MM-DD/raw/dark
   
   imagenes/INSTRUMENTO/YYYY-MM-DD/raw/flat
   
   imagenes/INSTRUMENTO/YYYY-MM-DD/raw/science
   
3) Colocar archivos .fits sin calibrar en las carpetas correspondientes (puede usar comando fitsheader para identificar imágenes).
4) Editar archivo <ARCHIVO>.par, donde <ARCHIVO> es cualquier nombre a definir (e.g., WASP-19b_TA.par).
5) Verificar que en archivo .par la información de la noche de observación a calibrar calce con la estructura de archivos en imágenes.
6) Ejecutar python calibracion.py <ARCHIVO>.par.
7) Ejecutar python prepare_photometry.py <ARCHIVO>.par.
8) Crear lista llamada "science_images.lst" con nombres de archivos .fits que serán incluídos en fotometría (en carpeta science dentro de imagenes "calibrated").
9) Ejecutar python apply_photometry.par.
10) Ver en carpeta "photometry" resultados de la fotometría (en particular archivo <ARCHIVO>.log que muestra rms por estrella de referencia y apertura).
11) Editar jupyter notebook correspondiente a primer analisis (e.g., jupyter notebook WASP-19b_TA.ipynb) y seleccionar estrella de referencia y apertura.
12) Editar jupyter notebook para realizar el fit MCMC (e.g., transitos_mcmc_XXXX.ipynb).
   
