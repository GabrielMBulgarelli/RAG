# Correcciones que más mejorarían el RAG

## Objetivo

Mejorar la calidad, trazabilidad y velocidad del flujo RAG local sin añadir modelos externos, servicios remotos, capacidades de producción ni un programa de evaluación nuevo. Las mejoras deben medirse reutilizando MultiHopRAG y las métricas existentes.

## Estado de referencia

El recuperador Hybrid constituye el baseline más fuerte, mientras el flujo Agentic añade coste sin mejorar la recuperación ni la respuesta:

| Métrica | Hybrid | Agentic |
|---|---:|---:|
| Recall@5 | 46.9% | 37.5% |
| MRR@5 | 58.8% | 29.2% |
| nDCG@5 | 45.3% | 29.8% |
| Cobertura de evidencia citada | — | 42.5% |
| Answer token F1 | — | 0.9% |
| Latencia P95 | 5,817 ms | 163,986 ms |
| Llamadas LLM por consulta | 0.0 | 4.5 |

La prioridad no es añadir más etapas agentic, sino hacer que cada etapa existente tenga una responsabilidad comprobable.

## 1. Calificación de evidencia por subconsulta

La calificación actual juzga el conjunto completo de fragmentos. Debe evaluar cada parte necesaria de la pregunta por separado.

Cambios propuestos:

- Registrar subconsultas respaldadas y no respaldadas.
- Distinguir evidencia suficiente, parcial, ausente y contradictoria.
- Permitir una respuesta completa sólo cuando todas las partes necesarias estén sustentadas.
- Pasar al generador exclusivamente los fragmentos marcados como relevantes.
- Conservar la relación entre fragmento, subconsulta y puntuaciones de recuperación.

Resultado esperado:

- Mayor cobertura de evidencia.
- Menos respuestas construidas con fragmentos irrelevantes.
- Abstenciones más justificadas y consistentes.

## 2. Validación determinista de respuestas y citas

La validación debe ir más allá de eliminar etiquetas desconocidas.

Debe detectar:

- etiquetas de cita inexistentes;
- afirmaciones factuales sin cita;
- citas a fragmentos no aprobados por el evaluador de evidencia;
- respuestas vacías;
- respuestas formadas únicamente por citas;
- fuentes mostradas pero no utilizadas en la respuesta.

La validación debe producir:

- texto saneado;
- fuentes realmente citadas;
- lista tipada de violaciones;
- estado final válido, limitado o abstención.

Se permitirá una sola reparación con el modelo generativo. Si la reparación vuelve a fallar, el sistema devolverá una respuesta limitada o se abstendrá.

## 3. Generación orientada a respuestas directas

La respuesta debe comenzar con una contestación breve y directa, seguida únicamente por la síntesis necesaria.

Reglas recomendadas:

- Una cita por cada afirmación factual verificable.
- No introducir información que no aparezca en los fragmentos aprobados.
- Expresar claramente cuando sólo una parte de la pregunta tiene respaldo.
- Evitar repetir extractos o describir el proceso interno del agente.
- Mantener `sources` limitado a las fuentes relevantes realmente citadas.

## 4. Control agentic más simple y determinista

Las decisiones LLM deben reservarse para los casos que realmente las necesitan.

Cambios propuestos:

- Resolver catálogo, preguntas simples evidentes y casos fuera de alcance mediante reglas deterministas.
- Usar Hybrid como recuperación documental predeterminada mientras siga superando las demás estrategias.
- Descomponer sólo preguntas comparativas o multihop.
- Ejecutar búsquedas independientes de subconsultas concurrentemente.
- Limitar el flujo a un máximo de un reintento dirigido.
- Usar temperatura `0` para routing, grading y generación reproducible.

Resultado esperado:

- Menor número de llamadas LLM.
- Menor latencia para preguntas simples.
- Decisiones más reproducibles.

## 5. Reintentos que preserven evidencia válida

Un reintento no debe sustituir todo lo recuperado previamente.

El flujo debe:

1. Identificar únicamente las subconsultas sin evidencia.
2. Generar nuevas consultas para esas partes.
3. Recuperar nuevos candidatos.
4. Fusionarlos con la evidencia válida anterior.
5. Volver a calificar la cobertura completa.

Esto evita perder fragmentos correctos y reduce trabajo repetido.

## 6. Recuperación adaptable y observable

La elección de estrategia debe responder al tipo de pregunta y a resultados medibles.

- Mantener Hybrid como baseline principal.
- Usar recuperación semántica aislada sólo cuando exista una ventaja clara.
- Reservar el flujo Agentic para preguntas que requieran composición de evidencia.
- Registrar por fragmento puntuaciones semántica, BM25, fusionada y de selección.
- Registrar duración, candidatos y llamadas por etapa en las trazas existentes.
- Mantener los detalles técnicos fuera de la respuesta principal, pero disponibles para inspección.

## 7. Resultados de evaluación comparables

La aplicación no debe interpretar el archivo más reciente como el mejor resultado de referencia si sólo contiene un smoke run parcial.

Cambios propuestos:

- Distinguir entre smoke run y benchmark estándar completo.
- Cargar por defecto el Full RAG Benchmark artifact completo más reciente.
- Mostrar claramente split, sistemas incluidos, cantidad de casos y fecha.
- Regenerar el Full RAG Benchmark artifact completo cuando se necesite evidencia actualizada.
- Mantener el benchmark de test como validación final, no como conjunto de ajuste.

## Orden recomendado

1. Evidencia por subconsulta y filtrado de contexto.
2. Validación tipada, reparación y fuentes exactas.
3. Reintentos con preservación de evidencia.
4. Rutas deterministas, concurrencia y temperatura cero.
5. Full RAG Benchmark artifact completo y comparación con el baseline.

## Criterios de éxito

- Agentic no queda por debajo de Hybrid en Recall@5.
- Cobertura de evidencia citada supera 42.5%.
- Precisión de citas llega a 100% después de validación.
- Answer token F1 mejora claramente sobre 0.9%.
- Exactitud de abstención alcanza al menos 85%.
- La latencia Agentic P95 disminuye al menos 50%.
- Las consultas simples evitan llamadas LLM innecesarias.
- Ningún reintento descarta evidencia válida ya recuperada.
