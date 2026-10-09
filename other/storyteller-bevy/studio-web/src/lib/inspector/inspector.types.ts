export type InspectorComponent
	= BevyReflect<"bevy_transform::components::transform::Transform", BevyTransform>
	| BevyReflect<"bevy_core::name::Name", BevyName>
	| BevyReflect<"bevy_pbr::pbr_material::StandardMaterial", StandardMaterial>
	;

export interface BevyReflect<K extends string, T> {
	name: K;
	value: T;
}

export interface BevyTransform {
	rotation: Quat;
	scale: Vec3;
	translation: Vec3;
}

export interface BevyName {
	hash: number;
	name: string;
}

export interface Quat {
	w: number;
	x: number;
	y: number;
	z: number;
}

export interface Vec3 {
	x: number;
	y: number;
	z: number;
}

export interface StandardMaterial {
	base_color: BevyColor;
	perceptual_roughness: number;
	metallic: number;
	reflectance: number;
}

export type BevyColor
	= Color_RgbaLinear
	| Color_Rgba
	;

export interface Color_RgbaLinear {
	RgbaLinear: Rgba;
}

export interface Color_Rgba {
	Rgba: Rgba;
}

export interface Rgba {
	red: number;
	green: number;
	blue: number;
	alpha: number;
}

export function isLinear(color: BevyColor): color is Color_RgbaLinear {
	return "RgbaLinear" in color;
}
