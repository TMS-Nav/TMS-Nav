import test from "node:test";
import assert from "node:assert/strict";
import { turbo, turboGradient, turboTable } from "../src/colormap.mjs";

const close = (a, b, tol = 0.02) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

test("turbo follows the reference table", () => {
  // rows of google's turbo lookup table. the polynomial drifts at the two ends, so
  // there it only has to be the right kind of colour: near black blue, dark red
  assert.ok(close(turbo(0.25), [0.15844, 0.73551, 0.92305], 0.05), `${turbo(0.25)}`);
  assert.ok(close(turbo(0.75), [0.98549, 0.54303, 0.11406], 0.05), `${turbo(0.75)}`);
  assert.ok(Math.max(...turbo(0)) < 0.25, `${turbo(0)}`);
  const [r, g, b] = turbo(1);
  assert.ok(r > 0.4 && r < 0.7 && g < 0.1 && b < 0.1, `${turbo(1)}`);
});

test("turbo runs blue to green to red and clamps", () => {
  const [r1, , b1] = turbo(0.15);
  const [, g2] = turbo(0.5);
  const [r3, , b3] = turbo(0.85);
  assert.ok(b1 > r1, "low end is blue");
  assert.ok(g2 > 0.8, "middle is green");
  assert.ok(r3 > b3, "high end is red");
  assert.deepEqual(turbo(-1), turbo(0));
  assert.deepEqual(turbo(2), turbo(1));
});

test("table and gradient come out the right shape", () => {
  const t = turboTable(16);
  assert.equal(t.length, 64);
  assert.equal(t[3], 255);
  assert.match(turboGradient(4), /^linear-gradient\(90deg, rgb\(\d+ \d+ \d+\) 0\.0%, .* 100\.0%\)$/);
});
