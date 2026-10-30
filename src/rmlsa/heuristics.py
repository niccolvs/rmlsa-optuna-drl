"""Algoritmos de asignación (allocators) con la firma que espera DREAM-ON GYM.

Firma: (src, dst, b, c, n, path, action) -> (Controller.Status, Connection)
    src, dst: nodos origen/destino.
    b: BitRate de la solicitud (lista de modulaciones con slots y alcance).
    c: Connection a la que se le agregan los enlaces/slots.
    n: Network (estado del espectro).
    path: path[src][dst] = lista de k rutas candidatas (listas de Link).
    action: decisión del agente (None durante el warm-up o en la heurística).
"""

from dreamongymv2.simNetPy import Controller


def _path_length(route):
    return sum(link.length for link in route)


def _modulation_slots(b, length):
    """Slots de la modulación más eficiente cuyo alcance cubre la ruta.

    Se asume que el archivo de bitrates lista las modulaciones de menor a
    mayor eficiencia (BPSK, QPSK, 8-QAM, 16-QAM), así que se recorre al revés.
    Retorna None si ninguna modulación alcanza.
    """
    for i in reversed(range(len(b.modulation))):
        if b.getReach(i) >= length:
            return b.getNumberofSlots(i)
    return None


def _assign_first_fit(route, slots_needed, c, n):
    """Busca el primer bloque contiguo libre en todos los enlaces de la ruta."""
    total_slots = n.getLink(0).getSlots()
    used = [False] * total_slots
    for link in route:
        link = n.getLink(link.id)
        for s in range(total_slots):
            used[s] = used[s] or link.getSlot(s)

    run, start = 0, 0
    for s in range(total_slots):
        if used[s]:
            run, start = 0, s + 1
            continue
        run += 1
        if run == slots_needed:
            for link in route:
                c.addLink(link, fromSlot=start, toSlot=start + run)
            return True
    return False


def _allocate_on_route(route, b, c, n):
    slots = _modulation_slots(b, _path_length(route))
    if slots is None:
        return False
    return _assign_first_fit(route, slots, c, n)


def first_fit(src, dst, b, c, n, path, action):
    """Ruta elegida por el agente (o la más corta si no hay acción) + First-Fit."""
    routes = path[src][dst]
    k = 0 if action is None else min(int(action), len(routes) - 1)
    if _allocate_on_route(routes[k], b, c, n):
        return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c


def spff(src, dst, b, c, n, path, action):
    """Shortest-Path First-Fit: ignora al agente, siempre usa la ruta 0.

    NSFNet_routes.json ordena las rutas de menor a mayor número de saltos.
    """
    if _allocate_on_route(path[src][dst][0], b, c, n):
        return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c
