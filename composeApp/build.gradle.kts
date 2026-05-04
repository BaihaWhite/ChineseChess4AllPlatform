import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.androidApplication)
}

repositories {
    google()
    mavenCentral()
}

kotlin {
    androidTarget {
        compilations.all {
            kotlinOptions {
                jvmTarget = "17"
            }
        }
    }

    jvm("desktop")

    sourceSets {
        commonMain.dependencies {
            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.ui)
            implementation(compose.components.resources)
        }

        androidMain.dependencies {
            implementation(libs.androidx.activity.compose)
            implementation(libs.androidx.core.ktx)
        }

        getByName("desktopMain").dependencies {
            implementation(compose.desktop.currentOs)
            implementation("org.jetbrains.skiko:skiko-awt-runtime-windows-x64:0.8.18")
            implementation("org.jetbrains.skiko:skiko-awt-runtime-macos-x64:0.8.18")
        }
    }
}

android {
    namespace = "com.chinesechess.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.chinesechess.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].manifest.srcFile("src/androidMain/AndroidManifest.xml")
    sourceSets["main"].res.srcDirs("src/androidMain/res")
}

compose.desktop {
    application {
        mainClass = "com.chinesechess.MainKt"
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb, TargetFormat.AppImage)
            packageName = "chinese-chess"
            packageVersion = "1.0.0"

            linux {
                iconFile.set(project.file("src/desktopMain/resources/icon-512.png"))
            }
            windows {
                iconFile.set(project.file("src/desktopMain/resources/icon.ico"))
            }
            macOS {
                iconFile.set(project.file("src/desktopMain/resources/icon-512.png"))
            }
        }
    }
}

gradle.projectsEvaluated {
    val desktopComp = project.kotlin.targets.getByName("desktop").compilations.getByName("main")
    project.tasks.withType<JavaExec>().matching { it.name == "desktopRun" }.configureEach {
        mainClass.set("com.chinesechess.MainKt")
        classpath = desktopComp.output.classesDirs + desktopComp.compileDependencyFiles + (desktopComp.runtimeDependencyFiles ?: project.files())
    }

    project.tasks.register<JavaExec>("generateOpeningBook") {
        group = "application"
        description = "Run self-play to generate the opening book. Customize with -PbookGames=500 -PbookPly=16 -PbookTopK=4"
        mainClass.set("com.chinesechess.engine.OpeningBookGenerator")
        classpath = desktopComp.output.classesDirs + desktopComp.compileDependencyFiles + (desktopComp.runtimeDependencyFiles ?: project.files())
        args = listOf(
            project.findProperty("bookGames")?.toString() ?: "300",
            project.findProperty("bookPly")?.toString() ?: "14",
            project.findProperty("bookTopK")?.toString() ?: "4",
            project.findProperty("bookOutput")?.toString() ?: "${project.projectDir}/src/commonMain/composeResources/files/opening_book.txt",
            project.findProperty("bookSave")?.toString() ?: "2000"
        )
    }
}
