// subject -> assets. ply meshes go through parcel as url assets, the targets
// json is imported directly so parcel hands us the parsed object. add a subject
// by dropping a folder under data and another block here
import T1_TARGETS from "./data/T1/targets.json";

export const DATASETS = {
  T1: {
    label: "T1",
    scalpURL: new URL("./data/T1/scalp.ply", import.meta.url),
    coilURL: new URL("./data/T1/coil.ply", import.meta.url),
    targets: T1_TARGETS,
  },
};
