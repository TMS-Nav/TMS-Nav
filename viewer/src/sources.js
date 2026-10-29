// the sources, licenses and disclaimer dialog. a native <dialog>, so the browser
// handles the focus trap, escape and handing focus back to the link that opened it
export function initSources() {
  const dialog = document.getElementById("sources");
  const open = document.getElementById("sources-open");
  if (!dialog || !open || typeof dialog.showModal !== "function") return;

  open.addEventListener("click", () => {
    dialog.showModal();
    // start at the top every time, and on the close button rather than the first link
    dialog.querySelector(".sources-body").scrollTop = 0;
    document.getElementById("sources-close").focus();
  });
  document.getElementById("sources-close").addEventListener("click", () => dialog.close());

  // a click on the dimmed page outside the box closes it. the box itself has no
  // padding, so any click that lands on the dialog element is on the backdrop
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close();
  });
}
