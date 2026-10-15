/**
 * Tiny software renderer used to inspect gaits from the command line.
 *
 * It writes uncompressed 24-bit BMP files, so no image library is needed and the
 * output can be opened directly. `gait-strip.mjs` uses it to draw a stroboscopic
 * film strip of a policy's episode, which is how the physics and the learned
 * gaits were checked during development.
 */

import { deflateSync } from 'node:zlib';

export class Bitmap {
  constructor(width, height, background = [16, 20, 28]) {
    this.width = width;
    this.height = height;
    this.px = Buffer.alloc(width * height * 3);
    this.clear(background);
  }

  clear([r, g, b]) {
    for (let i = 0; i < this.px.length; i += 3) {
      this.px[i] = r;
      this.px[i + 1] = g;
      this.px[i + 2] = b;
    }
  }

  /** Alpha blend a colour into a pixel (x, y with y down). */
  blend(x, y, color, alpha = 1) {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
    const i = ((y | 0) * this.width + (x | 0)) * 3;
    const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
    this.px[i] = this.px[i] * (1 - a) + color[0] * a;
    this.px[i + 1] = this.px[i + 1] * (1 - a) + color[1] * a;
    this.px[i + 2] = this.px[i + 2] * (1 - a) + color[2] * a;
  }

  fillRect(x0, y0, x1, y1, color, alpha = 1) {
    for (let y = Math.round(y0); y <= Math.round(y1); y++) {
      for (let x = Math.round(x0); x <= Math.round(x1); x++) this.blend(x, y, color, alpha);
    }
  }

  circle(cx, cy, r, color, alpha = 1) {
    const r2 = r * r;
    for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++) {
      for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
        const dx = x + 0.5 - cx;
        const dy = y + 0.5 - cy;
        if (dx * dx + dy * dy <= r2) this.blend(x, y, color, alpha);
      }
    }
  }

  /** A capsule: a thick line with round caps, i.e. exactly the collision shape. */
  capsule(x0, y0, x1, y1, r, color, alpha = 1) {
    const len = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(len / Math.max(0.5, r * 0.5)));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      this.circle(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, color, alpha);
    }
  }

  line(x0, y0, x1, y1, color, alpha = 1) {
    const len = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(len));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      this.blend(Math.round(x0 + (x1 - x0) * t), Math.round(y0 + (y1 - y0) * t), color, alpha);
    }
  }

  /**
   * Serialise as a PNG. Node's zlib does the compression, so a valid PNG is a
   * signature, three chunks and a CRC - no image library required.
   */
  toPNG() {
    const { width, height } = this;
    const raw = Buffer.alloc((width * 3 + 1) * height);
    for (let y = 0; y < height; y++) {
      raw[y * (width * 3 + 1)] = 0; // filter type: none
      this.px.copy
        ? this.px.copy(raw, y * (width * 3 + 1) + 1, y * width * 3, (y + 1) * width * 3)
        : raw.set(this.px.subarray(y * width * 3, (y + 1) * width * 3), y * (width * 3 + 1) + 1);
    }
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(width, 0);
    ihdr.writeUInt32BE(height, 4);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // colour type: truecolour
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    const idat = deflateSync(raw, { level: 6 });
    const chunks = [
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', idat),
      chunk('IEND', Buffer.alloc(0)),
    ];
    return Buffer.concat(chunks);
  }
}

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

/** Palette used for both the CLI renders and the browser client. */
export const COLORS = {
  trunk: [63, 127, 208],
  legNear: [90, 160, 232],
  legFar: [42, 95, 158],
  foot: [240, 180, 41],
  ground: [40, 46, 58],
  grid: [58, 68, 84],
  dark: [10, 12, 18],
};

/**
 * Draw one pose of the robot from a `snapshot()` of the environment.
 * @param {Bitmap} bmp
 * @param {{bodies:Array}} snapshot
 * @param {(x:number,y:number)=>[number,number]} project world -> pixels
 * @param {{alpha?:number, shade?:number, colors?:object}} opts
 */
export function drawSnapshot(bmp, snapshot, project, opts = {}) {
  const alpha = opts.alpha ?? 1;
  const shade = opts.shade ?? 1;
  const colors = opts.colors || COLORS;
  const tint = (c) => [c[0] * shade, c[1] * shade, c[2] * shade];
  for (const body of snapshot.bodies) {
    const key = body.g === 'trunk' ? 'trunk' : body.n.startsWith('foot') ? 'foot' : body.n.startsWith('thigh') || body.n.startsWith('shin') ? (body.g === 'near' ? 'legNear' : 'legFar') : 'trunk';
    const color = tint(colors[key] || [200, 200, 200]);
    // The radius has to be scaled, not projected: it is a length, not a position.
    for (const [ax, ay, bx, by, r] of body.s) {
      const [x0, y0] = project(ax, ay);
      const [x1, y1] = project(bx, by);
      const radiusPx = r * opts.scale;
      bmp.capsule(x0, y0, x1, y1, radiusPx, color, alpha);
    }
  }
}
