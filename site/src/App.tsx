import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { Pillars } from "./components/Pillars";
import { Penalties } from "./components/Penalties";
import { Method } from "./components/Method";
import { CodeStory } from "./components/CodeStory";
import { Metrics } from "./components/Metrics";
import { Install } from "./components/Install";
import { Footer } from "./components/Footer";
import { DocsPage } from "./pages/DocsPage";

export default function App() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  if (hash.startsWith("#/docs/")) {
    return <DocsPage slug={hash.slice("#/docs/".length)} />;
  }
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <Nav />
      <main id="main-content">
        <Hero />
        <Pillars />
        <Penalties />
        <Method />
        <CodeStory />
        <Metrics />
        <Install />
      </main>
      <Footer />
    </div>
  );
}
import { useEffect, useState } from "react";
