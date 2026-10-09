import { FormFieldBaseElement } from "@storyteller/studio-ui/forms/form-field-base";
import { PropertyValues, html, unsafeCSS } from "lit";
import { customElement, property } from "lit/decorators.js";
import { classMap } from "lit/directives/class-map.js";
import { styleMap } from "lit/directives/style-map.js";

import { type StyleOption, type VstStyle } from "./inspector-tabs/style-tab";

import styles from "./style-select.element.scss?inline";

@customElement("sts-style-select")
export class StyleSelect extends FormFieldBaseElement {
	static override styles = unsafeCSS(styles);

	@property({ reflect: true })
	override role = "listbox"

	@property({ attribute: false })
	options: readonly StyleOption[] = [];

	@property() value?: VstStyle;

	protected override update(changes: PropertyValues<this>): void {
		if (changes.has("value"))
			this.internals.setFormValue(this.value ?? "");

		super.update(changes);
	}

	protected override render = () => html`
	    <div>
			<!-- 
			TODO(bt): preview image
			<div
				class=${classMap({
					"bubble": true,
				})}
				style=${styleMap({
					"background-image": `url(https://fakeyou.com/images/landing/onboarding/styles/style-2d-anime.webp)`,
				})}
			>
				<div class="gradient"></div>
				<span class="label">
					${ "Foo" }
				</span>
			</div>
			-->
		    <select
				@change=${(event: InputEvent) => {
					const value = (event.target as HTMLSelectElement).value;

					// this.onChange({ target: { name: this.name, value: itemValue } });
					this.dispatchEvent(new CustomEvent("value-change", {
						detail: value,
					}))
				} }
			>
				${ this.options.map(({ label, value }) => html`
				    <option value="${value}">
					    ${ label }
					</option>`
				)}
		    </select>
		</div>
	`
}
