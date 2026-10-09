// Diagnostic: dump the active document's active text layer's `textKey` descriptor as text.
// Usage: run-jsx.sh dump-text.jsx [psd-path]  (opens the file if given, closes it afterwards)

function cTID(s) { return charIDToTypeID(s); }
function sTID(s) { return stringIDToTypeID(s); }
function idName(id) { var s = typeIDToStringID(id); return s ? s : typeIDToCharID(id); }

function dumpDesc(d, ind) {
    var out = [];
    for (var i = 0; i < d.count; i++) {
        var k = d.getKey(i);
        out.push(ind + idName(k) + ": " + dumpVal(d, k, d.getType(k), ind));
    }
    return out.join("\n");
}

function dumpVal(d, k, t, ind) {
    switch (t) {
        case DescValueType.OBJECTTYPE: return "{" + idName(d.getObjectType(k)) + "}\n" + dumpDesc(d.getObjectValue(k), ind + "  ");
        case DescValueType.LISTTYPE: return dumpList(d.getList(k), ind + "  ");
        case DescValueType.ENUMERATEDTYPE: return idName(d.getEnumerationType(k)) + "." + idName(d.getEnumerationValue(k));
        case DescValueType.UNITDOUBLE: return d.getUnitDoubleValue(k) + " " + idName(d.getUnitDoubleType(k));
        case DescValueType.DOUBLETYPE: return d.getDouble(k);
        case DescValueType.INTEGERTYPE: return d.getInteger(k);
        case DescValueType.BOOLEANTYPE: return d.getBoolean(k);
        case DescValueType.STRINGTYPE: return JSON_q(d.getString(k));
        default: return "<" + t + ">";
    }
}

function dumpList(l, ind) {
    var out = ["[" + l.count + "]"];
    for (var i = 0; i < l.count; i++) {
        var t = l.getType(i);
        if (t === DescValueType.OBJECTTYPE) out.push(ind + "- {" + idName(l.getObjectType(i)) + "}\n" + dumpDesc(l.getObjectValue(i), ind + "    "));
        else if (t === DescValueType.LISTTYPE) out.push(ind + "- " + dumpList(l.getList(i), ind + "  "));
        else if (t === DescValueType.DOUBLETYPE) out.push(ind + "- " + l.getDouble(i));
        else if (t === DescValueType.INTEGERTYPE) out.push(ind + "- " + l.getInteger(i));
        else if (t === DescValueType.UNITDOUBLE) out.push(ind + "- " + l.getUnitDoubleValue(i));
        else out.push(ind + "- <" + t + ">");
    }
    return out.join("\n");
}

function JSON_q(s) { return "\"" + String(s).replace(/\r/g, "\\r").replace(/\n/g, "\\n") + "\""; }

var opened = null;
if (typeof arguments !== "undefined" && arguments.length > 0 && arguments[0]) opened = app.open(new File(arguments[0]));
var r = new ActionReference();
r.putEnumerated(cTID("Lyr "), cTID("Ordn"), cTID("Trgt"));
var res;
try {
    res = dumpDesc(executeActionGet(r).getObjectValue(cTID("Txt ")), "");
} catch (e) {
    res = "error: " + e;
}
if (opened) opened.close(SaveOptions.DONOTSAVECHANGES);
res;
