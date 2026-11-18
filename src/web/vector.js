// Vector-display components: everything is drawn as stroked SVG paths,
// the way a beam would draw it. No fills, no raster text.

import { GLYPHS, CAP, BASE } from "./hershey.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const GLYPH_H = BASE - CAP;            // 21 units from cap line to baseline

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------

export function svg(tag, attrs = {}, children = [])
{
   const node = document.createElementNS(SVG_NS, tag);
   for(const [k, v] of Object.entries(attrs))
   {
      node.setAttribute(k, v);
   }
   for(const c of children)
   {
      node.appendChild(c);
   }
   return node;
}

export function el(tag, attrs = {}, children = [])
{
   const node = document.createElement(tag);
   for(const [k, v] of Object.entries(attrs))
   {
      if(k === "class")
      {
         node.className = v;
      }
      else if(k.startsWith("on"))
      {
         node.addEventListener(k.slice(2), v);
      }
      else if(v !== undefined && v !== null && v !== false)
      {
         node.setAttribute(k, v === true ? "" : v);
      }
   }
   for(const c of [].concat(children))
   {
      if(c === null || c === undefined || c === false)
      {
         continue;
      }
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
   }
   return node;
}

// ---------------------------------------------------------------------------
// Hershey stroke text
// ---------------------------------------------------------------------------

function textPath(str)
{
   let x = 0;
   let d = "";
   for(const ch of str)
   {
      const glyph = GLYPHS[ch] || GLYPHS["?"];
      if(glyph[1])
      {
         // Offset each glyph's coordinates by the running pen position
         d += glyph[1].replace(/([ML])(-?\d+) (-?\d+)/g,
            (_, cmd, gx, gy) => `${cmd}${+gx + x} ${gy}`);
      }
      x += glyph[0];
   }
   return { d, width: x };
}

// Text drawn with the beam. `size` is the cap-to-baseline height in px.
export function vtext(str, { size = 12, cls = "" } = {})
{
   const { d, width } = textPath(String(str));
   const pad = 3;
   const node = svg("svg", {
      class: `vtext ${cls}`.trim(),
      viewBox: `${-pad} ${CAP - pad} ${width + pad * 2} ${GLYPH_H + pad * 2}`,
      width: ((width + pad * 2) * size / GLYPH_H).toFixed(1),
      height: ((GLYPH_H + pad * 2) * size / GLYPH_H).toFixed(1),
      role: "img",
      "aria-label": str,
   });
   node.appendChild(svg("path", { d, "vector-effect": "non-scaling-stroke" }));
   return node;
}

// ---------------------------------------------------------------------------
// Icons (stroke-only paths on a 24x24 grid)
// ---------------------------------------------------------------------------

const ICONS = {
   plus: "M12 5V19M5 12H19",
   cross: "M6 6L18 18M18 6L6 18",
   dup: "M4 8H14V20H4ZM9 4H20V16",
   left: "M15 5L8 12L15 19",
   right: "M9 5L16 12L9 19",
   play: "M7 4L19 12L7 20Z",
   pause: "M8 5V19M16 5V19",
   prev: "M6 5V19M18 5L9 12L18 19Z",
   next: "M18 5V19M6 5L15 12L6 19Z",
   save: "M5 4H16L20 8V20H5ZM8 4V9H15V4M8 20V14H17V20",
   load: "M12 4V16M7 11L12 16L17 11M5 20H19",
   restart: "M19 12A7 7 0 1 1 15 5.7M15 2V6H19",
};

export function icon(name, size = 22)
{
   return svg("svg", { class: "icon", viewBox: "0 0 24 24", width: size, height: size,
                       "aria-hidden": "true" },
              [svg("path", { d: ICONS[name], "vector-effect": "non-scaling-stroke" })]);
}

export function iconButton(name, label, onClick, { disabled = false, cls = "" } = {})
{
   return el("button", { class: `ibtn ${cls}`.trim(), type: "button", "aria-label": label,
                         title: label, disabled, onclick: onClick }, [icon(name)]);
}

// A button whose caption is drawn in stroke text
export function textButton(caption, onClick, { cls = "", label = caption, disabled = false } = {})
{
   return el("button", { class: `tbtn ${cls}`.trim(), type: "button", "aria-label": label,
                         disabled, onclick: onClick }, [vtext(caption, { size: 11 })]);
}

// ---------------------------------------------------------------------------
// LED indicator
// ---------------------------------------------------------------------------

export function led(caption, cls = "")
{
   const lamp = svg("svg", { class: "lamp", viewBox: "0 0 16 16", width: 14, height: 14,
                             "aria-hidden": "true" },
                    [svg("circle", { cx: 8, cy: 8, r: 6 }),
                     svg("circle", { class: "core", cx: 8, cy: 8, r: 2.5 })]);
   const node = el("span", { class: `led ${cls}`.trim(), role: "status", "aria-label": caption },
                   [lamp, vtext(caption, { size: 9 })]);
   node.set = (on) => node.classList.toggle("on", !!on);
   return node;
}

// ---------------------------------------------------------------------------
// Two-position slide switch
// ---------------------------------------------------------------------------

export function slideSwitch(leftCap, rightCap, value, onChange, { label = "" } = {})
{
   const lever = svg("rect", { class: "lever", x: 3, y: 3, width: 18, height: 14, rx: 2 });
   const track = svg("svg", { class: "track", viewBox: "0 0 44 20", width: 44, height: 20,
                              "aria-hidden": "true" },
                     [svg("rect", { x: 0.75, y: 0.75, width: 42.5, height: 18.5, rx: 3 }), lever]);
   const left = vtext(leftCap, { size: 9, cls: "cap-left" });
   const right = vtext(rightCap, { size: 9, cls: "cap-right" });
   const node = el("button", { class: "slide", type: "button", role: "switch",
                               "aria-label": label || `${leftCap} / ${rightCap}` },
                   [left, track, right]);
   function show(v)
   {
      value = v;
      node.setAttribute("aria-checked", v ? "true" : "false");
      node.classList.toggle("right", v);
      lever.setAttribute("x", v ? 23 : 3);
   }
   node.addEventListener("click", () =>
   {
      show(!value);
      onChange(value);
   });
   show(value);
   return node;
}

// ---------------------------------------------------------------------------
// Waveform selector: each button draws its own waveform
// ---------------------------------------------------------------------------

function wavePath(kind)
{
   const pts = [];
   for(let i = 0; i <= 64; i++)
   {
      const t = i / 64;                       // two cycles across the button
      const p = (t * 2) % 1;
      let y;
      if(kind === "sine")
      {
         y = Math.sin(t * 4 * Math.PI);
      }
      else if(kind === "triangle")
      {
         y = 1 - 4 * Math.abs(p - 0.5);
      }
      else if(kind === "sawtooth")
      {
         y = 2 * p - 1;
      }
      else
      {
         y = p < 0.5 ? 1 : -1;
      }
      pts.push([4 + t * 40, 14 - y * 9]);
   }
   if(kind === "sawtooth" || kind === "square")
   {
      // Draw the vertical edges crisply instead of as slopes between samples
      const out = [];
      for(let i = 0; i < pts.length; i++)
      {
         if(i > 0 && Math.abs(pts[i][1] - pts[i - 1][1]) > 9)
         {
            out.push([pts[i][0], pts[i - 1][1]]);
         }
         out.push(pts[i]);
      }
      return "M" + out.map((p) => p.map((n) => n.toFixed(1)).join(" ")).join("L");
   }
   return "M" + pts.map((p) => p.map((n) => n.toFixed(1)).join(" ")).join("L");
}

export const WAVEFORMS = ["sine", "triangle", "sawtooth", "square"];

export function waveformSelector(value, onChange)
{
   const node = el("div", { class: "waves", role: "radiogroup", "aria-label": "Waveform" });
   const buttons = WAVEFORMS.map((kind) =>
   {
      const b = el("button", { class: "wave", type: "button", role: "radio",
                               "aria-label": kind, title: kind },
                   [svg("svg", { viewBox: "0 0 48 28", width: 48, height: 28, "aria-hidden": "true" },
                        [svg("path", { d: wavePath(kind), "vector-effect": "non-scaling-stroke" })])]);
      b.addEventListener("click", () =>
      {
         show(kind);
         onChange(kind);
      });
      node.appendChild(b);
      return [kind, b];
   });
   function show(v)
   {
      for(const [kind, b] of buttons)
      {
         b.classList.toggle("on", kind === v);
         b.setAttribute("aria-checked", kind === v ? "true" : "false");
      }
   }
   show(value);
   return node;
}

// ---------------------------------------------------------------------------
// Rotary knob
// ---------------------------------------------------------------------------
//
// opts: { label, min, max, step, log, values (discrete list), fmt, unit,
//         parse (text -> value or the raw text if unparseable) }
// Drag vertically to turn; arrow keys nudge; tap the readout to type.

const SWEEP = 270;                      // degrees of travel
const START = 135;                      // angle of the minimum, from +x clockwise
const TICKS = 27;

function polar(cx, cy, r, deg)
{
   const a = deg * Math.PI / 180;
   return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

export function knob(value, onChange, opts)
{
   const o = Object.assign({ min: 0, max: 1, step: 0.01, log: false, values: null,
                             fmt: (v) => String(v), label: "" }, opts);

   function toFrac(v)
   {
      if(typeof v !== "number" || !isFinite(v))
      {
         return 0;
      }
      let f;
      if(o.values)
      {
         let best = 0;
         o.values.forEach((x, i) =>
         {
            if(Math.abs(x - v) < Math.abs(o.values[best] - v))
            {
               best = i;
            }
         });
         f = best / (o.values.length - 1);
      }
      else if(o.log)
      {
         f = Math.log(v / o.min) / Math.log(o.max / o.min);
      }
      else
      {
         f = (v - o.min) / (o.max - o.min);
      }
      return Math.min(1, Math.max(0, f));
   }

   function fromFrac(f)
   {
      f = Math.min(1, Math.max(0, f));
      if(o.values)
      {
         return o.values[Math.round(f * (o.values.length - 1))];
      }
      let v = o.log ? o.min * Math.pow(o.max / o.min, f) : o.min + f * (o.max - o.min);
      const step = o.log ? Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))) - 2) : o.step;
      v = Math.round(v / step) * step;
      return +v.toPrecision(6);
   }

   // -- dial geometry --
   const cx = 32, cy = 32, r = 21;
   const ticks = [];
   const dialKids = [svg("circle", { class: "rim", cx, cy, r: r - 6 })];
   for(let i = 0; i < TICKS; i++)
   {
      const deg = START + SWEEP * i / (TICKS - 1);
      const [x1, y1] = polar(cx, cy, r, deg);
      const [x2, y2] = polar(cx, cy, r + (i % 13 === 0 ? 7 : 4), deg);
      const t = svg("line", { class: "tick", x1, y1, x2, y2 });
      ticks.push(t);
      dialKids.push(t);
   }
   const pointer = svg("line", { class: "pointer", x1: cx, y1: cy, x2: cx, y2: cy - r + 8 });
   dialKids.push(pointer);
   const dial = svg("svg", { class: "dial", viewBox: "0 0 64 64", width: 64, height: 64,
                             tabindex: 0, role: "slider", "aria-label": o.label }, dialKids);

   const readout = el("button", { class: "readout", type: "button",
                                  "aria-label": `Type a value for ${o.label}` });
   const node = el("div", { class: "knob" },
                   [dial, readout, o.label ? vtext(o.label, { size: 9, cls: "legend" }) : null]);

   function show(v)
   {
      value = v;
      const f = toFrac(v);
      const deg = START + SWEEP * f;
      const [x2, y2] = polar(cx, cy, r - 8, deg);
      pointer.setAttribute("x2", x2.toFixed(2));
      pointer.setAttribute("y2", y2.toFixed(2));
      ticks.forEach((t, i) => t.classList.toggle("lit", i / (TICKS - 1) <= f + 1e-9));
      readout.replaceChildren(vtext(typeof v === "number" ? o.fmt(v) : String(v), { size: 11 }));
      dial.setAttribute("aria-valuenow", typeof v === "number" ? v : "");
      dial.setAttribute("aria-valuetext", typeof v === "number" ? o.fmt(v) : String(v));
   }

   function commit(v)
   {
      if(v !== value)
      {
         show(v);
         onChange(v);
      }
   }

   // -- drag to turn --
   let drag = null;
   dial.addEventListener("pointerdown", (e) =>
   {
      drag = { y: e.clientY, f: toFrac(value), id: e.pointerId };
      dial.setPointerCapture(e.pointerId);
      node.classList.add("turning");
      e.preventDefault();
   });
   dial.addEventListener("pointermove", (e) =>
   {
      if(!drag || e.pointerId !== drag.id)
      {
         return;
      }
      commit(fromFrac(drag.f + (drag.y - e.clientY) / 240));
   });
   const end = () =>
   {
      drag = null;
      node.classList.remove("turning");
   };
   dial.addEventListener("pointerup", end);
   dial.addEventListener("pointercancel", end);

   // -- keyboard --
   dial.addEventListener("keydown", (e) =>
   {
      const n = o.values ? 1 / (o.values.length - 1) : 0.01;
      const delta = { ArrowUp: n, ArrowRight: n, ArrowDown: -n, ArrowLeft: -n,
                      PageUp: n * 10, PageDown: -n * 10 }[e.key];
      if(delta !== undefined)
      {
         commit(fromFrac(toFrac(value) + delta));
         e.preventDefault();
      }
   });

   // -- tap the readout to type an exact value --
   readout.addEventListener("click", () =>
   {
      const input = el("input", { class: "entry", type: "text", inputmode: "decimal",
                                  "aria-label": o.label,
                                  value: typeof value === "number" ? o.fmt(value) : String(value) });
      let done = false;
      const finish = (keep) =>
      {
         if(done)
         {
            return;
         }
         done = true;
         input.replaceWith(readout);
         if(keep)
         {
            const text = input.value.trim();
            const n = o.parse ? o.parse(text) : Number(text);
            // Unparseable text goes to the server as typed, so its error shows on the field
            commit(typeof n === "number" && isFinite(n) ? n : text);
         }
      };
      input.addEventListener("keydown", (e) =>
      {
         if(e.key === "Enter")
         {
            finish(true);
         }
         else if(e.key === "Escape")
         {
            finish(false);
         }
      });
      input.addEventListener("blur", () => finish(true));
      readout.replaceWith(input);
      input.focus();
      input.select();
   });

   show(value);
   node.set = show;
   return node;
}

// A knob that edits radians while showing and accepting degrees
export function phaseKnob(radians, onChange, label = "PHASE")
{
   const toDeg = (r) => +(r * 180 / Math.PI).toFixed(1);
   return knob(typeof radians === "number" ? toDeg(radians) : radians,
               (deg) => onChange(typeof deg === "number" ? deg * Math.PI / 180 : deg),
               { label, min: -180, max: 180, step: 1, fmt: (v) => `${v} DEG`,   // Hershey ASCII has no degree sign
                 parse: (t) => Number(t.replace(/deg|°/i, "").trim()) });
}
