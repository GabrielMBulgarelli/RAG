# Funcionalidades con mayor valor de producto

## Objetivo

Convertir el prototipo local en una aplicación RAG clara, demostrable y útil para un usuario real, manteniendo el flujo principal simple: añadir documentos, hacer una pregunta, revisar la respuesta y comprobar su evidencia.

Las funcionalidades se priorizan por valor visible, confianza y capacidad de demostrar competencias de IA aplicada. No se incluyen autenticación, despliegue, escalado ni otros elementos de producción.

## 1. Visor de evidencia y navegación de citas

Cada cita debe ser interactiva y permitir inspeccionar su origen sin entrar en detalles técnicos.

Experiencia propuesta:

- Seleccionar una cita desde la respuesta.
- Mostrar archivo, página, extracto y estado de relevancia.
- Resaltar la parte del extracto que sustenta la afirmación.
- Abrir una vista previa de la página cuando el formato lo permita.
- Mantener puntuaciones y trazas dentro de Technical details.

Valor:

- Hace visible la principal ventaja del proyecto: respuestas verificables.
- Permite detectar rápidamente respuestas incompletas o mal sustentadas.
- Produce una demostración de portafolio más convincente que una tabla de scores.

## 2. Ciclo de actualización documental

La aplicación debe distinguir entre añadir un documento nuevo y actualizar uno existente.

Funcionalidad propuesta:

- Detectar nombre, ruta lógica y hash del contenido.
- Advertir cuando el archivo ya existe sin cambios.
- Ofrecer actualizar cuando el contenido cambió.
- Reemplazar de forma segura los chunks anteriores sólo después de indexar correctamente la nueva versión.
- Evitar copias duplicadas de distintas revisiones del mismo documento.
- Mostrar fecha de indexación, versión y configuración utilizada.

Valor:

- Convierte el corpus en un recurso mantenible.
- Reduce resultados contradictorios producidos por versiones duplicadas.

## 3. Progreso, cancelación y estados de ejecución

Las operaciones largas deben explicar qué ocurre y permitir recuperar el control.

Se debe mostrar progreso para:

- carga de modelos;
- extracción y fragmentación;
- generación de embeddings;
- recuperación y generación de respuesta;
- reindexado y reconstrucción;
- evaluación.

La interfaz debe incluir:

- etapa actual expresada en lenguaje sencillo;
- conteo de documentos o casos procesados cuando esté disponible;
- duración transcurrida;
- acción de cancelación segura;
- resultado parcial o recuperación clara después de un error.

## 4. OCR local opcional

Los PDFs escaneados deben detectarse antes de producir un documento vacío o de baja calidad.

Flujo propuesto:

1. Extraer texto normalmente.
2. Detectar páginas sin texto suficiente.
3. Informar que el documento parece escaneado.
4. Permitir ejecutar OCR local de forma explícita.
5. Conservar página y procedencia para las citas.

El OCR debe ser opcional porque aumenta tiempo y dependencias.

## 5. Vista previa y ficha del documento

Al seleccionar un documento se debe mostrar una ficha compacta con:

- ruta relativa y tipo;
- número de páginas y chunks;
- fecha de indexación;
- estado de compatibilidad;
- modelo de embeddings y configuración de chunking;
- último error, si existe;
- acciones Update, Reindex y Delete.

Los IDs y hashes permanecerán en detalles técnicos, no en el flujo principal.

## 6. Workspace guiado por estado

La interfaz debe presentar una sola acción principal según el estado actual:

- Sin documentos: **Add and index files**.
- Documentos disponibles, modelos sin cargar: **Load AI models**.
- Sistema listo: compositor y **Ask**.
- Índice incompatible: acción concreta para repararlo.
- Respuesta disponible: revisar respuesta y evidencia.

Los estados técnicos secundarios deben permanecer colapsados. No deben aparecer simultáneamente múltiples advertencias que describan el mismo bloqueo.

## 7. Evaluation orientado a decisiones

La vista de evaluación debe responder rápidamente tres preguntas:

1. ¿Qué sistema recupera mejor?
2. ¿La respuesta está correctamente sustentada?
3. ¿Qué coste añade cada flujo?

Funcionalidad propuesta:

- Cargar el Full RAG Benchmark artifact completo más reciente.
- Diferenciar benchmark completo de smoke run.
- Mostrar comparación contra un baseline estable.
- Resaltar mejora, regresión o resultado sin cambio mediante texto e iconos.
- Mantener configuración, rutas y metadatos bajo Advanced options.
- Mostrar fallos agrupados por causa y permitir inspeccionar cada caso.

No se añaden datasets ni métricas nuevas; se mejora la presentación de las existentes.

## 8. Exportación útil y reproducible

Además de exportar la conversación, el archivo debe incluir:

- pregunta y respuesta final;
- fuentes citadas;
- estado de evidencia y validación;
- modelos y configuración relevantes;
- fecha;
- identificador de la sesión o ejecución.

La exportación técnica completa puede mantenerse separada de una versión legible para compartir.

## 9. Pulido para demostración y portafolio

El proyecto debe ser comprensible antes de ejecutar código.

Elementos recomendados:

- captura principal actualizada;
- demostración corta del flujo documento → pregunta → evidencia;
- diagrama sencillo de ingestión, recuperación y respuesta;
- explicación de decisiones y compromisos técnicos;
- Full RAG Benchmark artifact reproducible;
- archivo `LICENSE` explícito;
- README consistente con los controles disponibles;
- ejemplos de preguntas simples, multihop, sin respuesta y contradictorias.

## Orden recomendado

1. Visor de evidencia y navegación de citas.
2. Actualización segura de documentos y prevención de duplicados.
3. Progreso y cancelación de operaciones largas.
4. Workspace guiado por estado.
5. Evaluation con baseline comparable.
6. Vista previa documental y OCR opcional.
7. Exportación enriquecida y materiales de portafolio.

## Criterios de éxito

- Un usuario nuevo completa el flujo principal sin abrir System status.
- Toda fuente mostrada corresponde a una cita de la respuesta.
- Una cita permite llegar a su documento, página y extracto.
- Actualizar un archivo no crea una copia lógica duplicada.
- Las operaciones largas muestran progreso y pueden cancelarse sin corromper el índice.
- Un PDF sin texto produce una advertencia accionable.
- Evaluation distingue claramente un smoke run de un benchmark estándar.
- El README y las capturas coinciden con la interfaz actual.
