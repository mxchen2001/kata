const { workspace } = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const pythonPath = workspace.getConfiguration("kata").get("pythonPath", "python3");

  const serverOptions = {
    command: pythonPath,
    args: ["-m", "kata.lsp"],
  };

  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "kata" }],
  };

  client = new LanguageClient("kata-lsp", "Kata Language Server", serverOptions, clientOptions);
  client.start();
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };
