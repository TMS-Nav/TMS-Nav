import { Vector3, Triangle } from "three";

// Exact closest point on the displayed triangles, not just the nearest vertex.
// The viewer's PLY geometry is already in world RAS mm with no mesh transform.
export function closestScalpPoint(geometry, point) {
  const positions = geometry.getAttribute("position");
  const index = geometry.getIndex();
  const count = index ? index.count : positions.count;
  const triangle = new Triangle();
  const candidate = new Vector3();
  const best = new Vector3();
  const normal = new Vector3();
  let distance = Infinity;
  for (let i = 0; i < count; i += 3) {
    for (const [offset, vertex] of [[0, triangle.a], [1, triangle.b], [2, triangle.c]]) {
      vertex.fromBufferAttribute(positions, index ? index.getX(i + offset) : i + offset);
    }
    triangle.closestPointToPoint(point, candidate);
    const d = candidate.distanceToSquared(point);
    if (d < distance) {
      distance = d;
      best.copy(candidate);
      triangle.getNormal(normal);
    }
  }
  if (!Number.isFinite(distance)) throw new Error("Scalp has no triangles.");
  return { point: best, normal };
}

export function surfaceStep(geometry, point, direction, distance) {
  const contact = closestScalpPoint(geometry, point);
  const tangent = direction.clone().addScaledVector(contact.normal, -direction.dot(contact.normal));
  if (tangent.lengthSq() < 1e-8) return contact;
  return closestScalpPoint(geometry, contact.point.addScaledVector(tangent.normalize(), distance));
}
