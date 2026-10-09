import * as studio from "@storyteller/studio";
import { bind } from "@storyteller/framework";
import { LitElement, PropertyValues, type TemplateResult, html, unsafeCSS } from "lit";
import { customElement, property } from "lit/decorators.js";

import "@storyteller/studio-ui/forms/text-field";

import styles from "./name-inspector.element.scss?inline";

@customElement("sts-name-inspector")
export class NameInspectorElement extends LitElement {
	static override styles = unsafeCSS(styles);

	@property() value = "";

	protected override updated(changes: PropertyValues<this>): void {
		// FIXME: `Name` component needs special handling due to:
		//   1. Its `hash` field, which doesn't get updated by just directly
		//      mutating the `name` field
		//   2. Its use in our hierarchy. We'll need to add some logic to check
		//      for changed names on existing entities and notify the frontend so
		//      the hierarchy can reflect the new value.

		// if (changes.has("value")) {
		// 	studio.updateComponentField({
		// 		component: "bevy_core::name::Name",
		// 		reflectPath: "name",
		// 		value: this.value,
		// 	});
		// }

		super.updated(changes);
	}

	protected override render = (): TemplateResult => html`
		<sts-text-field .value=${bind(this, "value")}></sts-text-field>
	`;
}
