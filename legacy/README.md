# legacy/ — Herramientas de desarrollo archivadas

Estos archivos **no forman parte del pipeline ni del runtime** del Agente Financiero.
Se conservan aquí solo como referencia histórica.

| Archivo | Qué era |
|---|---|
| `_writer.py` | Herramienta de un solo uso que escribía el código fuente de otros módulos con rutas absolutas hardcodeadas. |
| `_write_opciones.py` | Variante de `_writer.py` para `opciones_derivados.py`. |
| `_test_streamlit.py` | Prueba trivial manual de Streamlit. |
| `technicals/` | Solo contiene `__pycache__/` con `.pyc` de un paquete `technicals` **cuyos fuentes ya no existen**. Las importaciones de `technicals.*` fallan en limpio; no resucitar sin reescribir los fuentes. |

No usar en producción ni volver a importarlos desde el código principal.
