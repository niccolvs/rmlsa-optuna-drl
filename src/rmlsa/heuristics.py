"""Allocators para DREAM-ON GYM: (src, dst, b, c, n, path, action) -> (status, connection)."""

from dreamongymv2.simNetPy import Controller


def path_length(route):
    return sum(link.length for link in route)


def modulation_slots(b, length):
    for i in reversed(range(len(b.modulation))):
        if b.getReach(i) >= length:
            return b.getNumberofSlots(i)
    return None


def used_slots(route, n):
    total_slots = n.getLink(0).getSlots()
    used = [False] * total_slots
    for link in route:
        link = n.getLink(link.id)
        for s in range(total_slots):
            used[s] = used[s] or link.getSlot(s)
    return used


def first_fit_start(used, slots_needed):
    run, start = 0, 0
    for s, busy in enumerate(used):
        if busy:
            run, start = 0, s + 1
            continue
        run += 1
        if run == slots_needed:
            return start
    return None


def _allocate_on_route(route, b, c, n):
    slots = modulation_slots(b, path_length(route))
    if slots is None:
        return False
    start = first_fit_start(used_slots(route, n), slots)
    if start is None:
        return False
    for link in route:
        c.addLink(link, fromSlot=start, toSlot=start + slots)
    return True


def first_fit(src, dst, b, c, n, path, action):
    routes = path[src][dst]
    k = 0 if action is None else min(int(action), len(routes) - 1)
    if _allocate_on_route(routes[k], b, c, n):
        return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c


def spff(src, dst, b, c, n, path, action):
    if _allocate_on_route(path[src][dst][0], b, c, n):
        return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c
