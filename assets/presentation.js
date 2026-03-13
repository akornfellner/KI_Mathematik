const THEME_MAP = {
  dark: "black",
  white: "white",
  hak: "white",
  htl: "white",
};

const LOGO_MAP = {
  hak: "hak.png",
  htl: "htl.png",
};

const HIGHLIGHT_MAP = {
  monokai:
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/monokai.css",
  github:
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github.min.css",
  atomOneLight:
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/atom-one-light.min.css",
  atomOneDark:
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/atom-one-dark.min.css",
};

const SEPARATOR_VERTICAL = String.raw`^\r?\n----\r?\n$`;
const SEPARATOR_NOTES = "^Note:";

const loadText = async (path) => {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.text();
};

const loadJson = async (path) => {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
};

const applyConfig = (config) => {
  const theme = THEME_MAP[config.theme] ?? THEME_MAP.dark;
  const highlightTheme = HIGHLIGHT_MAP[config.highlightTheme] ?? HIGHLIGHT_MAP.monokai;

  document.documentElement.lang = config.lang || "en";
  document.title = config.title || "Slides";
  document.body.classList.remove("theme-dark", "theme-white", "theme-hak", "theme-htl");
  const themeClass = ["white", "hak", "htl"].includes(config.theme) ? config.theme : "dark";
  document.body.classList.add(`theme-${themeClass}`);

  document.getElementById("reveal-theme").href =
    `https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/${theme}.css`;
  document.getElementById("highlight-theme").href = highlightTheme;
  document.getElementById("footer-center").textContent = config["footer-center"] || "";
};

const applyLogo = (config) => {
  const logo = document.getElementById("brand-logo");
  const logoFile = LOGO_MAP[config.theme];

  if (!logoFile) {
    logo.hidden = true;
    return;
  }

  logo.addEventListener("load", () => {
    logo.hidden = false;
  });
  logo.addEventListener("error", () => {
    logo.hidden = true;
  });

  logo.src = logoFile;
};

const buildSlides = async (root) => {
  const topicFiles = await loadJson("slides/topics/_index.json");

  // Title slide from slides/slides.md
  const titleSection = document.createElement("section");
  titleSection.setAttribute("data-markdown", "slides/slides.md");
  titleSection.setAttribute("data-separator-notes", SEPARATOR_NOTES);
  root.appendChild(titleSection);

  // One section per topic file — ---- separates vertical slides within
  for (const file of topicFiles) {
    const section = document.createElement("section");
    section.setAttribute("data-markdown", `slides/topics/${file}`);
    section.setAttribute("data-separator-vertical", SEPARATOR_VERTICAL);
    section.setAttribute("data-separator-notes", SEPARATOR_NOTES);
    root.appendChild(section);
  }
};

const enableLiveReload = () => {
  if (!window.EventSource) {
    return;
  }

  const events = new window.EventSource("/__events");
  events.onmessage = (event) => {
    if (event.data === "reload") {
      window.location.reload();
    }
  };
  events.onerror = () => {
    events.close();
  };
};

const main = async () => {
  const configText = await loadText("presentation.yml");

  const config = window.jsyaml.load(configText) ?? {};
  applyConfig(config);
  applyLogo(config);

  const root = document.getElementById("slides-root");
  await buildSlides(root);

  const deck = new window.Reveal({
    hash: true,
    controls: true,
    progress: true,
    center: true,
    transition: "slide",
    plugins: [window.RevealMarkdown, window.RevealHighlight, window.RevealNotes],
  });

  await deck.initialize();
  enableLiveReload();
};

main().catch((error) => {
  console.error(error);
  document.getElementById("slides-root").innerHTML = `
    <section>
      <h2>Presentation failed to load</h2>
      <pre>${error.message}</pre>
    </section>
  `;
});
