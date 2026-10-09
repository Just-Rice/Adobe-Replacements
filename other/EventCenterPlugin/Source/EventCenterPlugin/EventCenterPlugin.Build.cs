// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class EventCenterPlugin : ModuleRules
{
	public EventCenterPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(new string[] {});

		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...

				// NB(bt): Using the commercial BlueprintWebSocket plugin described in README.md
				//  Add this in the Unreal Engine Plugins menu. 
				//  If not present, use the Epic Games Launcher's Unreal Library Marketplace.
				"BlueprintWebSocket/Public"
				//"BlueprintWebSocket/Private",
			}
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...

				// NB(bt): Using the commercial BlueprintWebSocket plugin described in README.md
				//  Add this in the Unreal Engine Plugins menu. 
				//  If not present, use the Epic Games Launcher's Unreal Library Marketplace.
				"BlueprintWebSocket"
			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
	}
}
