---
name: ClustMetaLearn
colors:
  surface: '#f8f9ff'
  surface-dim: '#d0dbed'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e6eeff'
  surface-container-high: '#dee9fc'
  surface-container-highest: '#d9e3f6'
  on-surface: '#121c2a'
  on-surface-variant: '#424754'
  inverse-surface: '#27313f'
  inverse-on-surface: '#eaf1ff'
  outline: '#727785'
  outline-variant: '#c2c6d6'
  surface-tint: '#005ac2'
  primary: '#0058be'
  on-primary: '#ffffff'
  primary-container: '#2170e4'
  on-primary-container: '#fefcff'
  inverse-primary: '#adc6ff'
  secondary: '#006c49'
  on-secondary: '#ffffff'
  secondary-container: '#6cf8bb'
  on-secondary-container: '#00714d'
  tertiary: '#b61722'
  on-tertiary: '#ffffff'
  tertiary-container: '#da3437'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ad'
  on-tertiary-fixed: '#410004'
  on-tertiary-fixed-variant: '#930013'
  background: '#f8f9ff'
  on-background: '#121c2a'
  surface-variant: '#d9e3f6'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-md:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  title-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  title-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  code-md:
    fontFamily: monospace
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 20px
  margin-mobile: 16px
  margin-desktop: 48px
---

## Brand & Style
The design system is engineered for high-density machine learning operations, prioritizing clarity, technical precision, and analytical focus. It adopts a **Corporate Modern** aesthetic influenced by data science platforms like Hugging Face, characterized by a utilitarian "scientific" feel that remains approachable through a clean, light-themed interface.

The brand personality is intellectual and reliable. It evokes an emotional response of organized control over complex data. To achieve this, the design utilizes generous whitespace, a strict adherence to grid systems, and a professional color palette that emphasizes functional status indicators (success, error, processing) over decorative elements.

## Colors
The color strategy employs a high-clarity light mode to maximize legibility during long research sessions. 

- **Primary Blue (#3B82F6)** is reserved for interactive states, primary actions, and active navigation indicators.
- **Surface & Background** are subtly differentiated; a cool-toned light gray (#F8F9FA) provides the canvas, while pure white (#FFFFFF) defines the analytical containers (cards).
- **Semantic Colors** (Success Green and Error Red) are used strictly for model performance metrics and system status alerts.
- **Neutrals** follow a hierarchical scale, with #1F2937 providing high-contrast readability for headers and #6B7280 for metadata and supporting labels.

## Typography
This design system utilizes **Inter** for all UI elements to ensure maximum legibility across dense data displays. 

- **Data Displays:** Use `display-lg` for primary model metrics (e.g., Accuracy, Loss) to create a clear focal point.
- **Hierarchy:** Card headers use `title-md`, providing a clear anchor for grouped information.
- **Density:** The system defaults to `body-md` (14px) for most interface text to accommodate the high-density requirements of ML dashboards.
- **Technical Content:** For hyperparameter lists or log outputs, use a monospaced stack to maintain alignment and scientific character.

## Layout & Spacing
The layout follows a **Fixed-Fluid Hybrid** model. The main navigation is a fixed top bar (64px height), while the dashboard content utilizes a fluid grid that optimizes for wide-screen monitors common in developer environments.

- **Grid:** A 12-column grid is standard for desktop. Elements should snap to 4px increments (the base unit).
- **Breakpoints:**
  - **Mobile (<768px):** Single column, 16px margins.
  - **Tablet (768px - 1280px):** 6-column internal grid, 24px margins.
  - **Desktop (>1280px):** 12-column grid, max-width 1600px, 48px margins.
- **Density:** Use `sm` (8px) for internal component spacing and `lg` (24px) for spacing between major cards or sections.

## Elevation & Depth
Depth is communicated through **Tonal Layering** supplemented by soft, functional shadows. 

- **Level 0 (Background):** #F8F9FA.
- **Level 1 (Cards/Surface):** #FFFFFF with a subtle 1px border (#E5E7EB). This provides a crisp "laboratory" feel.
- **Shadow-sm:** `0 1px 2px 0 rgba(0, 0, 0, 0.05)`. Used for standard cards to provide a slight lift from the background.
- **Shadow-md:** `0 4px 6px -1px rgba(0, 0, 0, 0.1)`. Used for interactive hover states or dropdown menus to signify temporary prominence.

## Shapes
The shape language is structured and professional.
- **Buttons & Large Containers:** Use a `rounded-lg` (8px) radius to provide a modern, approachable feel without appearing overly casual.
- **Input Fields & Small Components:** Use a `rounded-md` (4px) radius. This tighter radius reflects precision and technical accuracy.
- **Status Pills:** Use a full pill-shape (999px) to clearly distinguish tags and status indicators from interactive buttons.

## Components
- **Buttons:** Primary buttons use a solid #3B82F6 background with white text. Secondary buttons use an outlined style with #E5E7EB border. Radius: 8px.
- **Cards:** White background, 1px #E5E7EB border, and `shadow-sm`. Card padding should be uniform at 24px.
- **Input Fields:** 1px border (#E5E7EB), 4px radius. On focus, the border transitions to #3B82F6 with a subtle blue outer glow.
- **Data Tables:** Clean, no vertical borders. Headers use `label-md` with muted text color (#6B7280) and 1px bottom border.
- **Top Navigation:** 64px height, white background, 1px bottom border. Nav items use 14px semi-bold text with 24px horizontal spacing.
- **Chips/Pills:** Used for ML tags (e.g., "NLP", "PyTorch"). Small padding (4px 12px), 12px font size, and light gray background.
- **Progress Indicators:** Linear bars for training progress using #3B82F6, with a subtle #E5E7EB track background.