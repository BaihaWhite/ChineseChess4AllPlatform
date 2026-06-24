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
            implementation("org.jetbrains.skiko:skiko-awt-runtime-linux-x64:0.8.18")
            implementation("org.jetbrains.skiko:skiko-awt-runtime-macos-x64:0.8.18")
        }
    }
}

android {
    namespace = "com.chinesechess.app"
    compileSdk = 35

    signingConfigs {
        create("release") {
            storeFile = file("chinese-chess.keystore")
            storePassword = "android123"
            keyAlias = "chinesechess"
            keyPassword = "android123"
            enableV1Signing = true
            enableV2Signing = true
            enableV3Signing = true
        }
    }

    defaultConfig {
        applicationId = "com.chinesechess.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "1.1.0"
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
        }
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
        jvmArgs("-Djava.library.path=${rootProject.file("engine/target/release")}")
    }
}

// Cross-platform uber-jar task
tasks.register<Jar>("packageCrossPlatformJar") {
    dependsOn("desktopJar")
    group = "compose desktop"
    description = "Create a cross-platform uber-jar with native libs for Linux and Windows"
    archiveBaseName.set("chinese-chess-crossplatform")
    archiveVersion.set("")
    archiveClassifier.set("")
    manifest {
        attributes("Main-Class" to "com.chinesechess.MainKt")
    }
    from(configurations.getByName("desktopRuntimeClasspath").map { if (it.isDirectory) it else zipTree(it) })
    from("build/classes/kotlin/desktop/main")
    from("build/processedResources/desktop/main")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

gradle.projectsEvaluated {
    val desktopComp = project.kotlin.targets.getByName("desktop").compilations.getByName("main")
    project.tasks.withType<JavaExec>().matching { it.name == "desktopRun" }.configureEach {
        mainClass.set("com.chinesechess.MainKt")
        classpath = desktopComp.output.classesDirs + desktopComp.compileDependencyFiles + (desktopComp.runtimeDependencyFiles ?: project.files())
    }
}
