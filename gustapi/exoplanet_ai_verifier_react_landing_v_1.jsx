import React, { useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * Exoplanet AI Verifier — single-file React page
 * Stack: Tailwind + Framer Motion
 *
 * Features
 * - Hero with 3D-ish animated planet (CSS only, lightweight)
 * - Interactive sections: Orbital Period, Planet Radius, Stellar Temperature
 * - Global consistency state (palette/patterns/atmosphere/starType/seed)
 * - Debounced image generation hooks (placeholders ready to connect to Images API)
 * - TESS Validator: posts to Flask /predict with three inputs
 * - Smooth, minimal, NASA-ish dark theme
 *
 * How to use
 * - Drop this component into a Next.js page (e.g., app/page.tsx or pages/index.tsx)
 * - Ensure Tailwind is configured
 * - Set BACKEND_URL (Flask) and IMAGE_API_FN to wire up real services
 */

// Smart backend resolution: try same-origin /api/predict, then env, then localhost
const DEFAULT_BACKEND_URL = (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_BACKEND_URL) || "http://127.0.0.1:5000/predict";

// Simple debounce hook
function useDebouncedCallback(fn, delay) {
  const t = useRef(null);
  return (...args) => {
    if (t.current) clearTimeout(t.current);
    t.current = setTimeout(() => fn(...args), delay);
  };
}

// Derive visual knobs from domain rules
function deriveOrbitalVisuals(days) {
  const size = days > 365 ? "massive" : days >= 10 ? "medium" : "small";
  const colorBase = days < 100 ? "orange-red" : days <= 365 ? "yellow-white" : "red giant";
  // Thickness proportional to days (normalize 0.5–10,000)
  const min = 0.5, max = 10000;
  const t = Math.min(1, Math.max(0, (days - min) / (max - min)));
  const thickness = 1 + Math.round(t * 12); // 1–13 px
  const intensity = 1 - t; // inverse
  return { size, colorBase, thickness, intensity: Number(intensity.toFixed(2)) };
}

function surfaceTypeByRadius(r) {
  if (r < 1.5) return "rocky, earth-like";
  if (r < 4) return "super-earth with thick atmosphere";
  if (r <= 15) return "gas giant with swirling clouds";
  return "super jupiter with storm systems";
}

function thermalFeaturesByTemp(K) {
  if (K < 3500) return "ice formations";
  if (K < 5000) return "temperate zones";
  if (K < 8000) return "lava flows";
  return "extreme radiation glow & scorched bands";
}

export default function ExoplanetAIVerifier() {
  // Global consistency state (critical)
  const [system, setSystem] = useState({
    baseColorPalette: ["#D97642", "#8B3A3A", "#4A6B5C", "#1a1d2e", "#6B4A7D"],
    surfacePatterns: "marbled bands with subtle storm cells",
    atmosphericStyle: "thin haze + auroral rim glow",
    starType: "G/K mix",
    generationSeed: "nasa-emc-2025"
  });

  // Interactive inputs
  const [orbitalDays, setOrbitalDays] = useState(4.124);
  const [planetRadius, setPlanetRadius] = useState(2.2);
  const [stellarTemp, setStellarTemp] = useState(5634);

  // Derived visuals
  const orbitalKnobs = useMemo(() => deriveOrbitalVisuals(orbitalDays), [orbitalDays]);
  const sizeLabel = useMemo(() => surfaceTypeByRadius(planetRadius), [planetRadius]);
  const thermalFeat = useMemo(() => thermalFeaturesByTemp(stellarTemp), [stellarTemp]);

  // Image cache (keyed by parameter combos)
  const [imgCache, setImgCache] = useState({});
  const [loadingKey, setLoadingKey] = useState(null);

  // Placeholder image generation (wire to your Images API here)
  const IMAGE_API_FN = async (promptKey, promptText) => {
    // TODO: replace with your DALL·E / MJ / Stability call
    // For now, return a placeholder URL constructed from the key
    await new Promise((r) => setTimeout(r, 800));
    return `https://placehold.co/1024x576/0a0a1a/e0e0e0?text=${encodeURIComponent(promptKey)}`;
  };

  const generateImage = useDebouncedCallback(async (kind, vars) => {
    const key = `${kind}_${Object.values(vars).join("_")}`;
    if (imgCache[key]) return; // cached

    setLoadingKey(key);

    // Build prompts per spec (shortened for clarity)
    let prompt = "";
    if (kind === "orbital") {
      const { DAYS, SIZE_VARIABLE, COLOR_BASE, THICKNESS, INTENSITY } = vars;
      prompt = `Photorealistic space scene: ${COLOR_BASE} star with ${SIZE_VARIABLE} corona, ` +
        `glowing orbital ring for ${DAYS} day period (thickness ${THICKNESS}, brightness ~${INTENSITY}). ` +
        `Dark space, NASA quality, cinematic.`;
    } else if (kind === "radius") {
      const { SIZE_MULTIPLIER } = vars;
      prompt = `Side-by-side comparison: Earth vs exoplanet ${SIZE_MULTIPLIER}×. ` +
        `Exoplanet inherits system colors ${system.baseColorPalette.join(",")} and patterns. ` +
        `Surface: ${surfaceTypeByRadius(SIZE_MULTIPLIER)}. Photorealistic, NASA quality.`;
    } else if (kind === "thermal") {
      const { TEMPERATURE } = vars;
      prompt = `Thermal viz for ${TEMPERATURE}K environment, gradients + ${thermalFeaturesByTemp(TEMPERATURE)}. ` +
        `Maintain size/patterns; cinematic NASA style.`;
    }

    const url = await IMAGE_API_FN(key, prompt);
    setImgCache((c) => ({ ...c, [key]: url }));
    setLoadingKey(null);
  }, 800);

  // Kick off default images on first render
  React.useEffect(() => {
    generateImage("orbital", {
      DAYS: orbitalDays,
      SIZE_VARIABLE: orbitalKnobs.size,
      COLOR_BASE: orbitalKnobs.colorBase,
      THICKNESS: orbitalKnobs.thickness,
      INTENSITY: orbitalKnobs.intensity,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // TESS validator call
  const [predLoading, setPredLoading] = useState(false);
  const [pred, setPred] = useState(null);
  const runPrediction = async () => {
    try {
      setPredLoading(true);
      setPred(null);
      const payload = {
        orbital_period: Number(orbitalDays),
        planet_radius: Number(planetRadius),
        stellar_temp: Number(stellarTemp),
      };

      // 1) Try same-origin API proxy (works in Next.js/Vercel without CORS)
      let res;
      try {
        res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`Proxy HTTP ${res.status}`);
      } catch (_proxyErr) {
        // 2) Fallback to direct Flask URL (env or localhost)
        res = await fetch(DEFAULT_BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      setPred(data.prediction);
    } catch (e) {
      setPred("ERROR");
      console.error(e);
    } finally {
      setPredLoading(false);
    }
  };

  // Small helper components
  const Section = ({ id, title, children }) => (
    <section id={id} className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h2 className="text-2xl sm:text-3xl font-semibold text-white mb-6">{title}</h2>
      <div className="glass rounded-2xl p-6 sm:p-8 border border-white/10 bg-white/5 shadow-xl">{children}</div>
    </section>
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-[#1a1d2e] via-[#0b0d17] to-black text-slate-200">
      {/* Starfield background */}
      <div className="pointer-events-none fixed inset-0 -z-10 opacity-40" style={{ backgroundImage: `radial-gradient(#ffffff22 1px, transparent 1px)`, backgroundSize: "3px 3px" }} />

      {/* HERO */}
      <header className="relative overflow-hidden">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20 sm:py-28">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="text-4xl sm:text-6xl font-bold">
                Discover New Worlds
              </motion.h1>
              <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }} className="mt-4 text-lg sm:text-xl text-slate-300">
                Explore exoplanets with AI trained on Kepler & K2 data
              </motion.p>
              <motion.a href="#orbital" whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }} className="inline-block mt-8 px-6 py-3 rounded-2xl bg-[#D97642] text-black font-semibold shadow-lg">
                Start Exploring
              </motion.a>
            </div>

            {/* CSS Planet */}
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6 }} className="relative mx-auto w-64 h-64 sm:w-80 sm:h-80">
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#6B4A7D] via-[#1a1d2e] to-black animate-[spin_20s_linear_infinite]" />
              <div className="absolute inset-0 rounded-full shadow-[0_0_60px_#6B4A7D66]" />
              <div className="absolute -inset-6 rounded-full blur-2xl opacity-40" style={{ background: "conic-gradient(from 90deg, #D97642, #8B3A3A, #4A6B5C, #6B4A7D, #1a1d2e)" }} />
            </motion.div>
          </div>
        </div>
      </header>

      {/* ORBITAL PERIOD */}
      <Section id="orbital" title="Orbital Period (days)">
        <div className="grid lg:grid-cols-5 gap-8">
          <div className="lg:col-span-2 space-y-4">
            <label className="block text-sm text-slate-300">{`Period: ${orbitalDays.toFixed(3)} days`}</label>
            <input
              type="range"
              min={0.5}
              max={10000}
              step={0.5}
              value={orbitalDays}
              onChange={(e) => {
                const v = Number(e.target.value);
                setOrbitalDays(v);
                const knobs = deriveOrbitalVisuals(v);
                generateImage("orbital", {
                  DAYS: v,
                  SIZE_VARIABLE: knobs.size,
                  COLOR_BASE: knobs.colorBase,
                  THICKNESS: knobs.thickness,
                  INTENSITY: knobs.intensity,
                });
                // update star type/color baseline once (or whenever cross a threshold)
                setSystem((s) => ({ ...s, starType: v > 365 ? "red giant" : v < 100 ? "K-type" : "G-type" }));
              }}
              className="w-full"
            />
            <div className="text-sm text-slate-400">Star size: <span className="text-slate-200 font-medium">{orbitalKnobs.size}</span> · Base color: <span className="font-medium">{orbitalKnobs.colorBase}</span> · Ring: <span className="font-medium">{orbitalKnobs.thickness}px</span> · Corona {orbitalKnobs.intensity}
            </div>
          </div>

          <div className="lg:col-span-3">
            <ImagePanel kind="orbital" vars={{
              DAYS: orbitalDays,
              SIZE_VARIABLE: orbitalKnobs.size,
              COLOR_BASE: orbitalKnobs.colorBase,
              THICKNESS: orbitalKnobs.thickness,
              INTENSITY: orbitalKnobs.intensity,
            }} imgCache={imgCache} loadingKey={loadingKey} />
            <InfoCard title="Science notes">
              Shorter periods often indicate planets close to their host star (possible hot Jupiters). Long periods suggest wide orbits and lower incident radiation.
            </InfoCard>
          </div>
        </div>
      </Section>

      {/* PLANET RADIUS */}
      <Section id="radius" title="Planet Radius (Earth = 1)">
        <div className="grid lg:grid-cols-5 gap-8 items-start">
          <div className="lg:col-span-2 space-y-4">
            <label className="block text-sm text-slate-300">{`Radius: ${planetRadius.toFixed(2)} × Earth`}</label>
            <input
              type="range"
              min={0.3}
              max={20}
              step={0.1}
              value={planetRadius}
              onChange={(e) => {
                const v = Number(e.target.value);
                setPlanetRadius(v);
                generateImage("radius", { SIZE_MULTIPLIER: v });
              }}
              className="w-full"
            />
            <div className="text-sm text-slate-400">Surface type: <span className="text-slate-200 font-medium">{sizeLabel}</span></div>
          </div>

          <div className="lg:col-span-3 grid sm:grid-cols-2 gap-6">
            <div className="rounded-xl border border-white/10 p-4 bg-black/30">
              <div className="mb-2 text-sm text-slate-400">Earth (ref)</div>
              <div className="aspect-[16/9] grid place-items-center bg-gradient-to-b from-sky-900/30 to-black/40 rounded-lg">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-sky-300 to-sky-800 shadow-[0_0_40px_#60a5fa55]" />
              </div>
            </div>
            <div className="rounded-xl border border-white/10 p-4 bg-black/30">
              <div className="mb-2 text-sm text-slate-400">Exoplanet ({planetRadius.toFixed(2)}×)</div>
              <ImagePanel kind="radius" vars={{ SIZE_MULTIPLIER: planetRadius }} imgCache={imgCache} loadingKey={loadingKey} />
            </div>
          </div>
        </div>
      </Section>

      {/* STELLAR TEMPERATURE */}
      <Section id="thermal" title="Stellar Temperature (K)">
        <div className="grid lg:grid-cols-5 gap-8 items-start">
          <div className="lg:col-span-2 space-y-4">
            <label className="block text-sm text-slate-300">{`Stellar Temperature: ${Math.round(stellarTemp)} K`}</label>
            <input
              type="range"
              min={2000}
              max={40000}
              step={50}
              value={stellarTemp}
              onChange={(e) => {
                const v = Number(e.target.value);
                setStellarTemp(v);
                generateImage("thermal", { TEMPERATURE: v });
              }}
              className="w-full"
            />
            <div className="text-sm text-slate-400">Thermal features: <span className="text-slate-200 font-medium">{thermalFeat}</span></div>
          </div>
          <div className="lg:col-span-3">
            <ImagePanel kind="thermal" vars={{ TEMPERATURE: stellarTemp }} imgCache={imgCache} loadingKey={loadingKey} />
            <InfoCard title="Thermal interpretation">
              Illumination and thermal gradients depend on stellar surface temperature and planet-star distance; look for halos and glow in environments &gt; 6000K.
            </InfoCard>
          </div>
        </div>
      </Section>

      {/* SUMMARY */}
      <Section id="summary" title="System Summary">
        <div className="grid md:grid-cols-2 gap-6 text-sm">
          <ul className="space-y-2">
            <li>Base palette: <span className="text-slate-300">{system.baseColorPalette.join(", ")}</span></li>
            <li>Surface patterns: <span className="text-slate-300">{system.surfacePatterns}</span></li>
            <li>Atmosphere: <span className="text-slate-300">{system.atmosphericStyle}</span></li>
          </ul>
          <ul className="space-y-2">
            <li>Star type: <span className="text-slate-300">{system.starType}</span></li>
            <li>Seed: <span className="text-slate-300">{system.generationSeed}</span></li>
            <li>Period/Radius/Temp: <span className="text-slate-300">{`${orbitalDays.toFixed(3)} d / ${planetRadius.toFixed(2)}× / ${Math.round(stellarTemp)} K`}</span></li>
          </ul>
        </div>
      </Section>

      {/* VALIDATOR (TESS) */}
      <Section id="validator" title="TESS Validator">
        <div className="grid md:grid-cols-3 gap-4 items-end">
          <LabeledInput label="Orbital Period (days)" value={orbitalDays} setValue={setOrbitalDays} step={0.001} min={0.5} max={10000} />
          <LabeledInput label="Planet Radius (Earth=1)" value={planetRadius} setValue={setPlanetRadius} step={0.01} min={0.3} max={20} />
          <LabeledInput label="Stellar Temp (K)" value={stellarTemp} setValue={setStellarTemp} step={1} min={2000} max={40000} />
        </div>
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={runPrediction} className="px-6 py-3 rounded-xl bg-emerald-500 text-black font-semibold">
            Analyze Candidate
          </motion.button>
          <span className="text-slate-400 text-sm">Use the same dark theme and visual consistency.
          </span>
        </div>
        <AnimatePresence>
          {predLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-6 flex items-center gap-3 text-slate-300">
              <div className="w-5 h-5 border-2 border-white/20 border-l-[#D97642] rounded-full animate-spin" />
              Analyzing transit signals...
            </motion.div>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {pred && !predLoading && (
            <motion.div initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ opacity: 0 }} className="mt-6 text-xl">
              {pred === "PLANET" && (
                <div className="text-emerald-400 drop-shadow-[0_0_12px_rgba(16,185,129,0.45)] font-semibold">Likely an Exoplanet</div>
              )}
              {pred === "FALSE POSITIVE" && (
                <div className="text-rose-400 drop-shadow-[0_0_12px_rgba(244,63,94,0.45)] font-semibold">Likely a False Positive</div>
              )}
              {pred === "ERROR" && (
                <div className="text-amber-400">Could not reach the model. Is Flask running at <code>{DEFAULT_BACKEND_URL}</code>?</div>
              )}
              {pred && pred !== "ERROR" && (
                <div className="text-slate-400 text-sm mt-1">Actual NASA Disposition: <span className="italic">(depends on the object)</span></div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </Section>

      <footer className="py-10 text-center text-xs text-slate-500">NASA Hackathon 2025 · Mérida (MX)</footer>
    </div>
  );
}

function ImagePanel({ kind, vars, imgCache, loadingKey }) {
  const key = `${kind}_${Object.values(vars).join("_")}`;
  const url = imgCache[key];
  const isLoading = loadingKey === key && !url;
  return (
    <div className="relative aspect-[16/9] rounded-xl overflow-hidden border border-white/10 bg-gradient-to-b from-slate-900/40 to-black/40 grid place-items-center">
      <AnimatePresence>
        {isLoading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-3 text-slate-300">
            <div className="w-6 h-6 border-2 border-white/20 border-l-[#D97642] rounded-full animate-spin" />
            Generating image...
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {url && (
          <motion.img key={key} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={url} alt={key} className="absolute inset-0 w-full h-full object-cover" />
        )}
      </AnimatePresence>
    </div>
  );
}

function InfoCard({ title, children }) {
  return (
    <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="text-sm font-semibold mb-2 text-slate-200">{title}</div>
      <div className="text-sm text-slate-300 leading-relaxed">{children}</div>
    </div>
  );
}

function LabeledInput({ label, value, setValue, step = 1, min, max }) {
  return (
    <label className="block text-sm">
      <span className="text-slate-300">{label}</span>
      <input type="number" value={value} step={step} min={min} max={max} onChange={(e) => setValue(Number(e.target.value))} className="mt-1 w-full px-3 py-2 rounded-xl bg-black/40 border border-white/10 focus:outline-none" />
    </label>
  );
}
