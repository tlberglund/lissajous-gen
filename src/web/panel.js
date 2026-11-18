// Control panel: edits the visualizer's config live. Every change is sent
// (debounced) as a complete document; the server validates it and the wall
// glides to it. Rejected fields are marked; the user's input is kept.

import { el, svg, vtext, icon, iconButton, textButton, led, slideSwitch,
         waveformSelector, knob, phaseKnob } from "./vector.js";

const SEND_DELAY = 250;
const AXES = { x: "CH1  X", y: "CH2  Y" };

const state = {
   doc: null,
   errors: [],
   restartRequired: false,
   selected: { x: 0, y: 0 },
   music: { source: "generator", tracks: [], current: null, playing: false },
   presets: [],
   presetChoice: null,
   confirmOverwrite: null,
};

const ui = {};          // long-lived elements (LEDs, section containers)

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

async function api(method, path, body)
{
   let resp;
   try
   {
      resp = await fetch(path, {
         method,
         headers: body !== undefined ? { "Content-Type": "application/json" } : {},
         body: body !== undefined ? JSON.stringify(body) : undefined,
         cache: "no-store",
      });
   }
   catch(e)
   {
      ui.link.set(false);
      throw e;
   }
   ui.link.set(true);
   let data = null;
   try
   {
      data = await resp.json();
   }
   catch(e)
   {
      data = null;
   }
   return { status: resp.status, body: data };
}

let sendTimer = null;
let sending = false;
let sendAgain = false;

function changed()
{
   clearTimeout(sendTimer);
   sendTimer = setTimeout(() =>
   {
      sendTimer = null;
      send();
   }, SEND_DELAY);
}

async function send()
{
   if(sending)
   {
      sendAgain = true;
      return;
   }
   sending = true;
   try
   {
      const { status, body } = await api("PUT", "/api/config", state.doc);
      if(status === 200)
      {
         state.errors = [];
         setRestart(body.restart_required);
         blink(ui.applied);
      }
      else if(body && body.errors)
      {
         state.errors = body.errors;
      }
      else
      {
         state.errors = [{ path: "", message: `server replied ${status}` }];
      }
      showErrors();
   }
   catch(e)
   {
      // LINK is already off; the next edit retries
   }
   finally
   {
      sending = false;
      if(sendAgain)
      {
         sendAgain = false;
         send();
      }
   }
}

function blink(lamp)
{
   lamp.set(true);
   clearTimeout(lamp.timer);
   lamp.timer = setTimeout(() => lamp.set(false), 350);
}

function setRestart(required)
{
   state.restartRequired = !!required;
   ui.restart.set(state.restartRequired);
   ui.restartBtn.hidden = !state.restartRequired;
}

// ---------------------------------------------------------------------------
// Document helpers
// ---------------------------------------------------------------------------

function sg()
{
   return state.doc.signal_generator;
}

function axisDoc(axis)
{
   return sg()[axis];
}

const RAND_RE = /^rand\(\s*([^,]*?)\s*,\s*([^)]*?)\s*\)$/;

function parseKV(v)
{
   if(typeof v === "string")
   {
      const m = RAND_RE.exec(v.trim());
      if(m)
      {
         const num = (s) => (s !== "" && isFinite(Number(s)) ? Number(s) : s);
         return { rand: true, min: num(m[1]), max: num(m[2]) };
      }
   }
   return { rand: false, v };
}

function formatKV(kv)
{
   return kv.rand ? `rand(${kv.min}, ${kv.max})` : kv.v;
}

function midpoint(v)
{
   const kv = parseKV(v);
   if(kv.rand)
   {
      const a = Number(kv.min), b = Number(kv.max);
      return isFinite(a) && isFinite(b) ? (a + b) / 2 : 1;
   }
   return typeof kv.v === "number" && kv.v > 0 ? kv.v : 1;
}

function fmtNum(v)
{
   if(typeof v !== "number")
   {
      return String(v);
   }
   if(Math.abs(v) >= 100)
   {
      return String(Math.round(v));
   }
   return String(+v.toPrecision(3));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

function friendly(path)
{
   if(!path)
   {
      return "CONFIG";
   }
   return path
      .replace("signal_generator.x", "CH1")
      .replace("signal_generator.y", "CH2")
      .replace(/\.keyframes\[(\d+)\]/, (_, i) => ` KEY ${+i + 1}`)
      .replace(/_color\[(\d)\]/, (_, i) => ` COLOR ${"RGB"[i]}`)
      .replace(/^(visualization|signal_generator|audio|display|playback|web)\./, "")
      .replace(/[._]/g, " ")
      .toUpperCase();
}

function matches(controlPath, errorPath)
{
   return errorPath === controlPath
      || errorPath.startsWith(controlPath + ".")
      || errorPath.startsWith(controlPath + "[");
}

function showErrors()
{
   ui.err.set(state.errors.length > 0);
   for(const node of document.querySelectorAll("[data-path]"))
   {
      const hit = state.errors.find((e) => matches(node.dataset.path, e.path));
      node.classList.toggle("bad", !!hit);
      node.title = hit ? hit.message : "";
   }
   ui.errors.replaceChildren(...state.errors.map((e) =>
      el("div", { class: "errline", role: "alert" },
         [vtext(`${friendly(e.path)}: ${e.message}`.toUpperCase(), { size: 10 })])));
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

function bezel(legend, children, { cls = "", note = null } = {})
{
   return el("section", { class: `bezel ${cls}`.trim(), "aria-label": legend },
             [el("header", { class: "legend" },
                 [vtext(legend, { size: 12 }), note ? vtext(note, { size: 9, cls: "note" }) : null]),
              ...children]);
}

// Knob bound to a document path like ["visualization", "glow_intensity"]
function boundKnob(path, opts, after = null)
{
   let target = state.doc;
   for(const k of path.slice(0, -1))
   {
      target = target[k];
   }
   const key = path[path.length - 1];
   const k = knob(target[key], (v) =>
   {
      target[key] = v;
      if(after)
      {
         after(v);
      }
      changed();
   }, opts);
   k.dataset.path = path.join(".").replace(/\.(\d+)$/, "[$1]");
   return k;
}

function colorGroup(caption, key)
{
   const color = state.doc.visualization[key];
   const swatch = svg("circle", { cx: 16, cy: 16, r: 11 });
   const paint = () => swatch.setAttribute("stroke",
      `rgb(${color.map((c) => Math.round(Math.min(1, Math.max(0, Number(c) || 0)) * 255)).join(",")})`);
   paint();
   const knobs = ["R", "G", "B"].map((ch, i) =>
      boundKnob(["visualization", key, i], { label: ch, min: 0, max: 1, step: 0.01, fmt: fmtNum }, paint));
   return el("div", { class: "group", "data-path": `visualization.${key}` },
             [el("div", { class: "group-head" },
                 [vtext(caption, { size: 10 }),
                  svg("svg", { class: "swatch", viewBox: "0 0 32 32", width: 28, height: 28,
                               "aria-hidden": "true" }, [swatch])]),
              el("div", { class: "row" }, knobs)]);
}

// ---------------------------------------------------------------------------
// SOURCE
// ---------------------------------------------------------------------------

function renderSource()
{
   const isMusic = !sg().enabled;
   const sw = slideSwitch("GEN", "MUSIC", isMusic, (music) =>
   {
      sg().enabled = !music;
      changed();
      renderSource();
      renderChannels();
      refreshMusicSoon();
   }, { label: "Source: generator or music" });

   const body = [el("div", { class: "row center" }, [sw])];

   if(!isMusic)
   {
      body.push(el("div", { class: "row" }, [
         boundKnob(["signal_generator", "transition_time"],
                   { label: "GLIDE", min: 0, max: 10, step: 0.1, fmt: (v) => `${fmtNum(v)} S` }),
      ]));
   }
   else
   {
      body.push(renderTracks());
      const m = state.music;
      body.push(el("div", { class: "row center transport" }, [
         iconButton("prev", "Previous track", () => transport("prev")),
         m.playing
            ? iconButton("pause", "Pause", () => transport("pause"))
            : iconButton("play", "Play", () => transport("resume")),
         iconButton("next", "Next track", () => transport("next")),
      ]));
      body.push(el("div", { class: "row" }, [
         el("div", { class: "switchcell", "data-path": "playback.auto_advance" }, [
            slideSwitch("STOP", "NEXT", state.doc.playback.auto_advance, (v) =>
            {
               state.doc.playback.auto_advance = v;
               changed();
            }, { label: "At end of track: stop or play next" }),
            vtext("TRACK END", { size: 9, cls: "legend" }),
         ]),
         boundKnob(["audio", "amplitude_scale"],
                   { label: "GAIN", min: 0, max: 2, step: 0.01, fmt: fmtNum }),
      ]));
   }
   ui.source.replaceChildren(bezel("SOURCE", body));
   showErrors();
}

function renderTracks()
{
   const m = state.music;
   if(m.tracks.length === 0)
   {
      return el("div", { class: "empty" }, [vtext("NO MP3 FILES IN THE MUSIC FOLDER", { size: 10 })]);
   }
   return el("ol", { class: "tracks" }, m.tracks.map((name) =>
      el("li", {}, [el("button", {
         class: `track ${name === m.current ? "current" : ""}`.trim(),
         type: "button", "aria-label": `Play ${name}`,
         "aria-current": name === m.current ? "true" : null,
         onclick: () => playTrack(name),
      }, [vtext(name === m.current ? ">" : " ", { size: 11, cls: "marker" }),
          vtext(name.replace(/\.mp3$/i, ""), { size: 11 })])])));
}

async function playTrack(name)
{
   await api("POST", "/api/music/play", { track: name }).catch(() => null);
   sg().enabled = false;
   refreshMusicSoon();
}

async function transport(action)
{
   await api("POST", `/api/music/${action}`).catch(() => null);
   refreshMusicSoon();
}

// Loading a track can take the Pi a second or two; look twice
function refreshMusicSoon()
{
   setTimeout(refreshMusic, 400);
   setTimeout(refreshMusic, 1800);
}

async function refreshMusic()
{
   try
   {
      const { status, body } = await api("GET", "/api/music");
      if(status === 200)
      {
         const before = JSON.stringify(state.music);
         state.music = body;
         if(before !== JSON.stringify(body) && !sg().enabled)
         {
            renderSource();
         }
      }
   }
   catch(e)
   {
      // LINK shows the problem
   }
}

// ---------------------------------------------------------------------------
// CH1 / CH2 keyframe editors
// ---------------------------------------------------------------------------

function renderChannels()
{
   for(const axis of Object.keys(AXES))
   {
      renderAxis(axis);
   }
}

function renderAxis(axis)
{
   const a = axisDoc(axis);
   const n = a.keyframes.length;
   state.selected[axis] = Math.min(state.selected[axis], n - 1);
   const sel = state.selected[axis];
   const base = `signal_generator.${axis}`;

   const waves = waveformSelector(a.waveform, (w) =>
   {
      a.waveform = w;
      changed();
   });
   waves.dataset.path = `${base}.waveform`;

   const strip = el("div", { class: "strip" });
   const drawStrip = () =>
   {
      strip.replaceChildren(...a.keyframes.map((kf, i) =>
      {
         const random = parseKV(kf.duration).rand;
         return el("button", {
            class: `seg ${i === sel ? "sel" : ""} ${random ? "random" : ""}`.trim(),
            type: "button",
            style: `flex-grow:${midpoint(kf.duration)}`,
            "aria-label": `Keyframe ${i + 1}`,
            "aria-pressed": i === sel ? "true" : "false",
            "data-path": `${base}.keyframes[${i}]`,
            onclick: () =>
            {
               state.selected[axis] = i;
               renderAxis(axis);
            },
         }, [vtext(String(i + 1), { size: 10 })]);
      }));
   };
   drawStrip();

   const kfs = a.keyframes;
   const edit = (fn) =>
   {
      fn();
      // A second keyframe makes durations mandatory; make sure they exist
      for(const kf of kfs)
      {
         if(kf.duration === undefined)
         {
            kf.duration = 1;
         }
      }
      changed();
      renderAxis(axis);
   };
   const tools = el("div", { class: "row tools" }, [
      iconButton("left", "Move keyframe earlier", () => edit(() =>
      {
         [kfs[sel - 1], kfs[sel]] = [kfs[sel], kfs[sel - 1]];
         state.selected[axis] = sel - 1;
      }), { disabled: sel === 0 }),
      iconButton("right", "Move keyframe later", () => edit(() =>
      {
         [kfs[sel + 1], kfs[sel]] = [kfs[sel], kfs[sel + 1]];
         state.selected[axis] = sel + 1;
      }), { disabled: sel === n - 1 }),
      iconButton("plus", "Add keyframe after this one", () => edit(() =>
      {
         kfs.splice(sel + 1, 0, JSON.parse(JSON.stringify(kfs[sel])));
         state.selected[axis] = sel + 1;
      })),
      iconButton("dup", "Duplicate keyframe at the end", () => edit(() =>
      {
         kfs.push(JSON.parse(JSON.stringify(kfs[sel])));
         state.selected[axis] = kfs.length - 1;
      })),
      iconButton("cross", "Delete keyframe", () => edit(() =>
      {
         kfs.splice(sel, 1);
         state.selected[axis] = Math.max(0, sel - 1);
      }), { disabled: n === 1 }),
   ]);

   const kf = kfs[sel];
   const fields = [
      ["frequency", "FREQ", { min: 0.1, max: 50, log: true, fmt: (v) => `${fmtNum(v)} HZ` }],
      ["phase", "PHASE", null],
      ["amplitude", "AMP", { min: 0, max: 1, step: 0.01, fmt: fmtNum }],
      ["duration", "TIME", { min: 0.1, max: 120, log: true, fmt: (v) => `${fmtNum(v)} S` }],
   ];
   const editor = el("div", { class: "kf-editor" }, fields.map(([field, caption, opts]) =>
      fieldRow(kf, field, caption, opts, `${base}.keyframes[${sel}].${field}`, () =>
      {
         if(field === "duration")
         {
            drawStrip();
            showErrors();
         }
      }, () => renderAxis(axis))));

   const gen = sg().enabled;
   const node = bezel(AXES[axis], [
      waves,
      el("div", { class: "strip-head" }, [vtext("TIMEBASE", { size: 9, cls: "legend" }),
                                          vtext(`KEY ${sel + 1} OF ${n}`, { size: 9, cls: "legend" })]),
      strip,
      tools,
      editor,
   ], { cls: gen ? "" : "idle", note: gen ? null : "GENERATOR OFF" });
   ui[axis].replaceChildren(node);
   showErrors();
}

// One keyframe field: a FIX/RND switch and one knob (fixed) or two (min/max)
function fieldRow(kf, field, caption, opts, path, onValue, rerender)
{
   const kv = parseKV(kf[field]);
   const make = (value, label, set) =>
   {
      if(field === "phase")
      {
         return phaseKnob(value, set, label);
      }
      return knob(value, set, Object.assign({}, opts, { label }));
   };
   const put = () =>
   {
      kf[field] = formatKV(kv);
      onValue();
      changed();
   };

   const sw = slideSwitch("FIX", "RND", kv.rand, (rand) =>
   {
      if(rand)
      {
         kv.min = kv.max = kv.v;
      }
      else
      {
         kv.v = typeof kv.min === "number" ? kv.min : 0;
      }
      kv.rand = rand;
      put();
      rerender();
   }, { label: `${caption}: fixed or random` });

   const knobs = kv.rand
      ? [make(kv.min, `${caption} MIN`, (v) => { kv.min = v; put(); }),
         make(kv.max, `${caption} MAX`, (v) => { kv.max = v; put(); })]
      : [make(kv.v, caption, (v) => { kv.v = v; put(); })];

   return el("div", { class: "field", "data-path": path },
             [el("div", { class: "switchcell" }, [sw]), el("div", { class: "row" }, knobs)]);
}

// ---------------------------------------------------------------------------
// BEAM (visualization)
// ---------------------------------------------------------------------------

function renderBeam()
{
   const v = ["visualization"];
   ui.beam.replaceChildren(bezel("BEAM", [
      colorGroup("TRACE", "trace_color"),
      el("div", { class: "row" }, [
         boundKnob([...v, "trace_width"], { label: "WIDTH", min: 0.5, max: 30, step: 0.5, fmt: fmtNum }),
         boundKnob([...v, "beam_sharpness"], { label: "FOCUS", min: 0.5, max: 10, step: 0.25, fmt: fmtNum }),
         boundKnob([...v, "decay_rate"], { label: "DECAY", min: 0, max: 10, step: 0.1, fmt: fmtNum }),
      ]),
      el("div", { class: "row" }, [
         boundKnob([...v, "glow_radius"], { label: "GLOW SIZE", min: 0, max: 30, step: 0.5, fmt: fmtNum }),
         boundKnob([...v, "glow_intensity"], { label: "GLOW", min: 0, max: 10, step: 0.1, fmt: fmtNum }),
      ]),
      colorGroup("BACKGROUND", "background_color"),
   ]));
}

// ---------------------------------------------------------------------------
// PRESETS
// ---------------------------------------------------------------------------

async function refreshPresets()
{
   try
   {
      const { status, body } = await api("GET", "/api/presets");
      if(status === 200)
      {
         state.presets = body.presets;
         renderPresets();
      }
   }
   catch(e)
   {
      // LINK shows the problem
   }
}

function renderPresets()
{
   const list = el("ol", { class: "presets" }, state.presets.map((name) =>
      el("li", {}, [el("button", {
         class: `preset ${name === state.presetChoice ? "current" : ""}`.trim(),
         type: "button", "aria-pressed": name === state.presetChoice ? "true" : "false",
         onclick: () =>
         {
            state.presetChoice = name;
            nameInput.value = name;
            state.confirmOverwrite = null;
            renderPresets();
         },
      }, [vtext(name.replace(/^config-/, "").toUpperCase(), { size: 11 })])])));

   const nameInput = el("input", { class: "entry wide", type: "text", placeholder: "preset name",
                                   "aria-label": "Preset name", value: state.presetChoice || "",
                                   autocapitalize: "off", spellcheck: "false" });
   const confirming = state.confirmOverwrite !== null;
   const status = el("div", { class: "presetmsg", role: "status" });

   ui.presets.replaceChildren(bezel("PRESETS", [
      list,
      el("div", { class: "row" }, [
         textButton("LOAD", () => loadPreset(state.presetChoice),
                    { disabled: !state.presetChoice, label: "Load the selected preset" }),
      ]),
      el("div", { class: "row save" }, [
         nameInput,
         textButton(confirming ? "OVERWRITE" : "SAVE",
                    () => savePreset(nameInput.value.trim(), status),
                    { cls: confirming ? "warn" : "", label: confirming ? "Confirm overwrite" : "Save as preset" }),
      ]),
      status,
   ]));
   if(confirming)
   {
      status.replaceChildren(vtext(`${state.confirmOverwrite.toUpperCase()} EXISTS. PRESS OVERWRITE TO REPLACE IT.`,
                                   { size: 9, cls: "warn" }));
   }
}

async function loadPreset(name)
{
   if(!name)
   {
      return;
   }
   const { status, body } = await api("POST", `/api/presets/${encodeURIComponent(name)}/load`);
   if(status === 200)
   {
      state.doc = body.config;
      state.errors = [];
      setRestart(body.restart_required);
      blink(ui.applied);
      renderAll();
   }
   else
   {
      state.errors = body && body.errors ? body.errors
         : [{ path: "", message: (body && body.error) || `could not load ${name}` }];
      showErrors();
   }
}

async function savePreset(name, status)
{
   if(!name)
   {
      status.replaceChildren(vtext("TYPE A NAME FIRST", { size: 9, cls: "warn" }));
      return;
   }
   const overwrite = state.confirmOverwrite === name;
   const { status: code, body } = await api("POST", `/api/presets/${encodeURIComponent(name)}`,
                                            { overwrite });
   if(code === 409)
   {
      state.confirmOverwrite = name;
      state.presetChoice = name;
      renderPresets();
      return;
   }
   state.confirmOverwrite = null;
   if(code === 200)
   {
      state.presets = body.presets;
      state.presetChoice = name;
      renderPresets();
      ui.presets.querySelector(".presetmsg")
         .replaceChildren(vtext(`SAVED ${name.toUpperCase()}`, { size: 9 }));
   }
   else
   {
      renderPresets();
      ui.presets.querySelector(".presetmsg")
         .replaceChildren(vtext(((body && body.error) || "save failed").toUpperCase(), { size: 9, cls: "warn" }));
   }
}

// ---------------------------------------------------------------------------
// SYSTEM (restart-required settings)
// ---------------------------------------------------------------------------

function renderSystem()
{
   const d = state.doc;
   const res = d.display.resolution;
   const auto = res === "auto";
   const wh = auto ? [1280, 720] : res;

   const numInput = (value, label, onSet) =>
   {
      const input = el("input", { class: "entry", type: "text", inputmode: "numeric",
                                  "aria-label": label, value: String(value) });
      input.addEventListener("change", () =>
      {
         const t = input.value.trim();
         onSet(/^\d+$/.test(t) ? Number(t) : t);
         changed();
      });
      return input;
   };

   const w = numInput(wh[0], "Width in pixels", (v) => { if(!auto) { d.display.resolution[0] = v; } });
   const h = numInput(wh[1], "Height in pixels", (v) => { if(!auto) { d.display.resolution[1] = v; } });
   w.disabled = h.disabled = auto;

   ui.system.replaceChildren(bezel("SYSTEM", [
      el("div", { class: "row" }, [
         el("div", { class: "switchcell", "data-path": "display.resolution" }, [
            slideSwitch("AUTO", "SET", !auto, (set) =>
            {
               d.display.resolution = set ? [Number(w.value) || 1280, Number(h.value) || 720] : "auto";
               changed();
               renderSystem();
            }, { label: "Resolution: automatic or set" }),
            el("div", { class: "row tight" }, [w, vtext("X", { size: 9 }), h]),
            vtext("RESOLUTION", { size: 9, cls: "legend" }),
         ]),
         el("div", { class: "switchcell", "data-path": "display.fullscreen" }, [
            slideSwitch("WINDOW", "FULL", d.display.fullscreen, (v) =>
            {
               d.display.fullscreen = v;
               changed();
            }, { label: "Fullscreen" }),
            vtext("SCREEN", { size: 9, cls: "legend" }),
         ]),
      ]),
      el("div", { class: "row" }, [
         boundKnob(["audio", "sample_rate"], { label: "SAMPLE RATE",
                   values: [22050, 32000, 44100, 48000, 96000], fmt: (v) => `${v / 1000}K` }),
         boundKnob(["audio", "buffer_size"], { label: "BUFFER",
                   values: [256, 512, 1024, 2048, 4096, 8192], fmt: String }),
         el("div", { class: "switchcell", "data-path": "web.port" }, [
            numInput(d.web.port, "Web port", (v) => { d.web.port = v; }),
            vtext("WEB PORT", { size: 9, cls: "legend" }),
         ]),
      ]),
   ], { cls: "system", note: "APPLIES ON RESTART" }));
   showErrors();
}

// ---------------------------------------------------------------------------
// Restart
// ---------------------------------------------------------------------------

async function restart()
{
   ui.restartBtn.disabled = true;
   await api("POST", "/api/restart").catch(() => null);
   ui.link.set(false);
   const started = Date.now();
   await new Promise((r) => setTimeout(r, 1500));
   while(Date.now() - started < 60000)
   {
      try
      {
         const { status } = await api("GET", "/api/status");
         if(status === 200)
         {
            break;
         }
      }
      catch(e)
      {
         // still restarting
      }
      await new Promise((r) => setTimeout(r, 1000));
   }
   ui.restartBtn.disabled = false;
   await loadAll();
}

// ---------------------------------------------------------------------------
// Startup and refresh
// ---------------------------------------------------------------------------

function renderAll()
{
   renderSource();
   renderChannels();
   renderBeam();
   renderPresets();
   renderSystem();
   showErrors();
}

async function loadAll()
{
   try
   {
      const [cfg, music, presets] = await Promise.all([
         api("GET", "/api/config"), api("GET", "/api/music"), api("GET", "/api/presets")]);
      state.doc = cfg.body.config;
      setRestart(cfg.body.restart_required);
      state.music = music.body;
      state.presets = presets.body.presets;
      state.errors = [];
      renderAll();
   }
   catch(e)
   {
      ui.errors.replaceChildren(el("div", { class: "errline", role: "alert" },
         [vtext("CANNOT REACH THE VISUALIZER. RETRYING.", { size: 10 })]));
      setTimeout(loadAll, 3000);
   }
}

function buildShell()
{
   ui.link = led("LINK");
   ui.applied = led("APPLIED");
   ui.err = led("ERR", "error");
   ui.restart = led("RESTART", "amber");
   ui.restartBtn = textButton("RESTART", restart, { cls: "warn", label: "Restart the visualizer now" });
   ui.restartBtn.hidden = true;
   ui.errors = el("div", { class: "errors" });
   for(const name of ["source", "x", "y", "beam", "presets", "system"])
   {
      ui[name] = el("div", { class: `slot slot-${name}` });
   }

   document.getElementById("panel").replaceChildren(
      el("header", { class: "top" }, [
         el("h1", {}, [vtext("LISSAJOUS", { size: 22, cls: "title" })]),
         el("div", { class: "leds" }, [ui.link, ui.applied, ui.err, ui.restart, ui.restartBtn]),
      ]),
      ui.errors,
      el("main", { class: "grid" },
         [ui.source, ui.presets, ui.x, ui.y, ui.beam, ui.system]),
   );
}

buildShell();
loadAll();

// Keep LINK honest and the music list current
setInterval(async () =>
{
   try
   {
      await api("GET", "/api/status");
      if(state.doc && !sg().enabled)
      {
         refreshMusic();
      }
   }
   catch(e)
   {
      // LINK is off
   }
}, 4000);

// Another client may have changed things while this tab was hidden
document.addEventListener("visibilitychange", () =>
{
   if(document.visibilityState === "visible" && !sending && !sendTimer)
   {
      loadAll();
   }
});
