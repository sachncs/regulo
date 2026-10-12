import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Nav } from "./Nav";

describe("Nav", () => {
  it("opens and closes the mobile navigation", () => {
    render(<Nav />);
    const button = screen.getByRole("button", { name: "Open navigation" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    fireEvent.click(screen.getAllByRole("link", { name: "Method" })[1]);
    expect(document.querySelector("#mobile-nav")).not.toBeInTheDocument();
  });
});
