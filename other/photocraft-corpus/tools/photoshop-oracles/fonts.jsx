// List fonts Photoshop sees that the generator may use (diagnostic).
var out = [];
for (var i = 0; i < app.fonts.length; i++) {
    var f = app.fonts[i];
    var n = f.postScriptName;
    if (/Nerd/.test(n)) continue;
    if (/Inter|JetBrainsMono-|Helvetica|Times|Georgia|Arial|SourceSans|SourceSerif|Minion|Myriad|Noto|DejaVu|Roboto|Courier|Menlo|Hiragino/i.test(n))
        out.push(n + " | " + f.family + " | " + f.style);
}
out.join("\n");
