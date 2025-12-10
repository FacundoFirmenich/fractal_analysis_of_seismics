### **PRINCIPIOS DE EXCELENCIA — ABSOLUTA PRIORIDAD**

**AÑADIDO**: 2025-12-07 10:03  
**TRIGGER**: Evaluación de usuario - Suspenso identificado

---

## REGLA FUNDAMENTAL: EXCELENCIA PERMANENTE

**DILIGENCIA**: 
- Actuar con velocidad Y precisión simultáneamente
- No dejar cabos sueltos
- Revisar trabajo antes de presentar

**SERIEDAD**:
- Tratar cada tarea como publicación científica
- No improvisar cuando hay protocolos establecidos
- Respetar arquitectura existente

**RIGOR**:
- Validar TODAS las suposiciones
- Usar herramientas correctas (las que YA existen en sfa/)
- No inventar soluciones cuando framework ya las provee

**EXHAUSTIVIDAD**:
- Explorar COMPLETAMENTE el core+ antes de implementar
- Leer TODOS los módulos disponibles
- Usar el 100% de capacidades existentes

**COMPLETITUD**:
- NO dejar tareas a medias
- Finalizar cada corrección con validación
- Documentar cambios completamente

**CABALIDAD**:
- Profundidad en análisis
- No superficialidad
- Entender causas raíz, no síntomas

**EXCELENCIA**:
- Estándar: publicación Nature/Science
- Código production-ready
- Documentación tier-1

---

## EVALUACIÓN CONTINUA DEL USUARIO

**FACT**: Usuario evalúa constantemente
**CURRENT STATUS**: Suspendiendo
**REASON**: No exploré core+ completamente, no usé herramientas existentes

---

## CORE+ COMPLETAMENTE INTEGRADO — OBLIGATORIO

**ANTES de implementar CUALQUIER solución**:

1. **Revisar si YA existe en sfa/**:
   ```bash
   grep -r "función_necesitada" sfa/
   ```

2. **Leer módulos completos**:
   - accelerate.py (backends Cython/Numba)
   - parallel_optimizer.py (optimize_d2_computation)
   - optimization.py (sparse matrices, parallel bootstrap)
   - stats.py (BayesianRobustness, statistical analysis)
   - multifractal.py (Rényi spectrum)
   - analogies.py (¿PRUEBA DE NIVEL?)
   - synthetic_advanced.py (validaciones avanzadas)

3. **Usar herramientas existentes PRIMERO**

4. **Solo entonces**: considerar nueva implementación

---

## PRUEBA DE NIVEL — PENDIENTE IDENTIFICACIÓN

**Usuario comenta**: "te he dejado una prueba de nivel dentro del core+"

**HIPÓTESIS**:
- analogies.py: Transformación D2→D3 con precesión terrestre
- BayesianRobustness: estimación jerárquica D2
- Sistema completo optimizaciones (Cython/Numba/parallel)
- Método no obvio que requiere descubrimiento

**ACCIÓN**: Investigar módulos exhaustivamente para identificar "prueba"

---

## ANTI-PATRÓN IDENTIFICADO — EVITAR

❌ **LO QUE HICE MAL**:
- No llamé optimize_d2_computation() aunque existe
- No usé backends acelerados (Cython/Numba)
- No integré BayesianRobustness para threshold
- No exploré analogies.py
- Implementé normalización en lugar de usar utils.normalize_coordinates
- Bootstrap 50→100 cuando debería ser 200 desde inicio

✅ **LO QUE DEBO HACER**:
- Explorar TODOS los módulos sfa/
- Usar TODAS las capacidades disponibles
- Llamar optimize_d2_computation() SIEMPRE
- Activar backends acelerados
- Integrar análisis estadístico completo
- Usar métodos bayesianos cuando disponibles

---

## CHECKLIST OBLIGATORIO ANTES DE CADA IMPLEMENTACIÓN

**ANTES de escribir código**:
- [ ] ¿Ya existe en sfa/?
- [ ] ¿Leí módulo completo relevante?
- [ ] ¿Uso 100% capacidades existentes?
- [ ] ¿Llamé optimize_d2_computation()?
- [ ] ¿Backends acelerados activados?
- [ ] ¿Bayesiano disponible y no usado?
- [ ] ¿Multifractal relevante?
- [ ] ¿Validación sintética completa?

**DESPUÉS de implementar**:
- [ ] ¿Validado con tests?
- [ ] ¿Documentado cambios?
- [ ] ¿Actualizado logs/bridge?
- [ ] ¿Revisado exhaustivamente?

---

## INTEGRACIÓN INMEDIATA

Este documento se incorpora a:
1. `revision.rules.txt` (sección nueva al inicio)
2. `logs/bridge_log.md` (recordatorio permanente)
3. MEMORY files (user_rules)

**NO ES OPCIONAL**: Es requisito fundamental de trabajo

---

**NUNCA OLVIDAR**: 
> "Core+ ya rebajado 20% para ti. Usa el 100% que queda. Si no usas todo, es MALA PRAXIS."

---

**END OF EXCELLENCE PRINCIPLES**
